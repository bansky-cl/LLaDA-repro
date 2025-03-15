import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    # 参考 ICLR2025 论文：https://arxiv.org/pdf/2409.02908
    # 5.2 介绍了32位的精度问题
    # 5.3 CATEGORICAL SAMPLING WITH TRUNCATED GUMBEL 介绍了一种采样的trick
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    # 输入的是一个block长度的mask序列，需要执行steps步降噪
    # 通过计算mask数量，给出一个值，表示当前降噪到什么程度了，说白就是噪声水平用mask比例去衡量

    # 计算batch内每个（block）序列有多少个mask
    mask_num = mask_index.sum(dim=1, keepdim=True)
    
    base = mask_num // steps
    remainder = mask_num % steps

    # 创建一个全0的tensor，大小为[batch, steps]，表示需要降噪的mask数量
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    # ?
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        <EOS> 126081
    '''
    # x =  [bsz, seqlen], 后几行全是基于seqlen的batch操作

    # 第一步，先拼起来，得到一个全mask的序列，总长度 L + gen_length (定长)
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    # 第二步，把prompt作为序列的前L个token input(条件)
    x[:, :prompt.shape[1]] = prompt.clone()
    # 第三步，标记batch内每个序列prompt和response的位置
    prompt_index = (x != mask_id)
    # [:, True ... False ...]

    # 计算block数
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length # 4 blocks

    # 计算采样步数
    assert steps % num_blocks == 0
    steps = steps // num_blocks # 32 steps

    # 其实这里有个问题？sample steps大于gen_length怎么办？
    # steps = gen_length，每步还原一个token
    # steps < gen_length，分块
    # steps > gen_length  ？论文里面没说研究这一块

    for num_block in range(num_blocks): # # block 内 remasking
        # 第四步，标记当前block需要操作的位置，即当前block与下一个block的范围 [batch, L + 块数1 * 块长: L + 块数2 * 块长]
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)

        # 计算当前block需要降噪的mask数量? 表示当前降噪程度？类似diffusBERT的做法
        # block长度的tensor，一开始全1，用于后面的remask策略
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        # 第五步，对block内需要操作的位置，进行batch形式的step步的采样（即扩散模型inference步数T）
        for i in range(steps): # <-- 主要复杂度来源，降噪的T步困难
            mask_index = (x == mask_id) # 其他block设为true，prompt和当前block是false
            # 分类器引导，其实就是条件生成
            if cfg_scale > 0.:
                un_x = x.clone()
                # 将prompt也mask掉？为什么
                un_x[prompt_index] = mask_id
                # [batch, prompt | MASK .. | MASK ..]
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            # 用trick+优化精度
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)

            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            # x0 是 [batch，prompt || 还原序列] 的 token id tensor

            if remasking == 'low_confidence':
                # 将logits通过softmax归一化转成概率probs
                # 即 bf16 的 [4.8125,  ..., -2.4219] 归一成 float64 的 [2.6659e-08, ..., 1.9231e-11]
                p = F.softmax(logits.to(torch.float64), dim=-1)
                # 取x0对应在p上的probs，形状也是b，l，表示每个位置的置信度（0-1）四位小数
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            # 确保在block内操作
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x) # 确保在block内操作
            confidence = torch.where(mask_index, x0_p, -np.inf) # 每次这个置信度tensor在block内都会多一个-inf即remask功能

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i]) # 确定每步需要remask的位置
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def main():
    device = 'cuda'

    mpath = f"/{...}/models/LLaDA-8B-Instruct"

    model = AutoModel.from_pretrained(pretrained_model_name_or_path=mpath, trust_remote_code=True, 
                                      torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=mpath, 
                                              trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    # token'<|startoftext|><|start_header_id|>user<|end_header_id|>\n\nLily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    # list[126080, 126346, 3840, 126347, 198, 198, 86059, 560, 2001, 220, 16, 17, 44137, 854, 6984, 352, 220, 19, 3871, 13, 4474, 378, 11, 1285, 9660, 220, 21, 44137, 854, 6984, 13, 2071, 1494, 44137, 560, 1285, 2001, 296, 220, 23, 3871, 30, 126348, 126346, 598, 10450, 126347, 198, 198]
    # token id 220 很奇怪？是空格吗，感觉模型很喜欢预测这个220
    input_ids = tokenizer(prompt)['input_ids']
    # tensor
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    # 假设不需要分块
    out = generate(model, input_ids, steps=32, gen_length=16, block_length=8, temperature=0., cfg_scale=0., remasking='low_confidence')

    # out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')


    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()
