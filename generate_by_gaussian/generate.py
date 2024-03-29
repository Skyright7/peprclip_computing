# 生成部分 (高斯噪声)
# 输入：csv文件（文件内容是作为基底用的种子多肽的氨基酸序列）
# 输出：csv文件（文件内容是生成的候选多肽的氨基酸序列）

import pandas as pd
import torch
import esm
from tqdm import tqdm
import random


import argparse

parser = argparse.ArgumentParser(description='generating model of Gaussian version.')

parser.add_argument('-data_path',type=str, required=True)
parser.add_argument('-num_base_peps',type=int, required=True)
parser.add_argument('-num_peps_per_base',type=int, required=True)
parser.add_argument('-min_length',type=int, required=True)
parser.add_argument('-max_length',type=int, required=True)
parser.add_argument('-sample_variances_down',type=int, required=True)
parser.add_argument('-sample_variances_up',type=int, required=True)
parser.add_argument('-sample_variances_step',type=int, required=True)
parser.add_argument('-output_path',type=str, required=True)

args = parser.parse_args()

# 主要参数
base_file_path = args.data_path
num_base_peps = args.num_base_peps
num_peps_per_base = args.num_peps_per_base
min_length = args.min_length
max_length = args.max_length
sample_variances_down = args.sample_variances_down
sample_variances_up = args.sample_variances_up
sample_variances_step = args.sample_variances_step
sample_variances = range(sample_variances_down,sample_variances_up, sample_variances_step)
output_path = args.output_path


# # 主要参数
# base_file_path = './data/Noisy_Dataset.csv'
# # peps_per_target = 8 # 每个sample为底生成几个候选多肽
# num_base_peps = 100 # 在候选集中随机选出几个多肽作为基底
# num_peps_per_base = 1000 # 每个sample为底生成几个候选多肽
# min_length = 15 # 生成的多肽的最短长度
# max_length = 18 # 生成的多肽的最长长度
# sample_variances = range(5,22, 4) # 设置高斯噪音的方差范围（这个值越低，生成的跟原样本就越像，越高，越不像）


def generate_pep_dict(base_file_path,num_base_peps,num_peps_per_base,min_length,max_length,sample_variances):
    n = num_peps_per_base * num_base_peps # 最终生成数
    # 加载数据
    sample_df = pd.read_csv(base_file_path)
    # 加载编码用模型
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.model.to(device)
    # -- 生成算法部分 --
    # 先依照多肽设置的长短范围，将符合这个长度范围的能用作sample的数据从原库中筛出
    base_peptides = sample_df.loc[(sample_df['pep_len'] <= max_length) & (sample_df['pep_len'] >= min_length)].pep_seq.to_list()
    # 随机采样100个作为基底
    sampled_peptides = random.sample(base_peptides, num_base_peps)
    generated_peptides = []
    for pep in tqdm(sampled_peptides):
        target_seq = pep
        batch_labels, batch_strs, batch_tokens = batch_converter([("target_seq", target_seq)])
        batch_tokens = batch_tokens.cuda()
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33].cpu()
        del batch_tokens
        num_samples_per_base = int(int(n / num_base_peps) / len(sample_variances))
        for i in sample_variances:
            for j in range(num_samples_per_base):
                gen_pep = token_representations + torch.randn(
                    token_representations.shape) * i * token_representations.var()
                aa_toks = list("ARNDCEQGHILKMFPSTWYV")
                aa_idxs = [alphabet.get_idx(aa) for aa in aa_toks]
                aa_logits = model.lm_head(gen_pep.cuda())[:, :, aa_idxs]
                predictions = torch.argmax(aa_logits, dim=2).tolist()[0]
                generated_pep_seq = "".join([aa_toks[i] for i in predictions])
                generated_peptides.append(generated_pep_seq[1:-1])

    # -- 生成算法部分结束 --
    return generated_peptides

out_list = generate_pep_dict(base_file_path, num_base_peps, num_peps_per_base, min_length, max_length, sample_variances)
df = pd.DataFrame(out_list,columns=['generated_peptides'])
df.to_csv(output_path)


# if __name__ == '__main__':
#     base_file_path = './data/generate_demo_data/input/Noisy_Dataset.csv'
#     num_base_peps = 100  # 在候选集中随机选出几个多肽作为基底
#     num_peps_per_base = 1000  # 每个sample为底生成几个候选多肽
#     min_length = 15  # 生成的多肽的最短长度
#     max_length = 18  # 生成的多肽的最长长度
#     sample_variances = range(5, 22, 4)  # 设置高斯噪音的方差范围（这个值越低，生成的跟原样本就越像，越高，越不像）
#     output_path = './data/generate_demo_data/output/generated_peptides.csv'
#     out_list = generate_pep_dict(base_file_path, num_base_peps, num_peps_per_base, min_length, max_length, sample_variances)
#     df = pd.DataFrame(out_list,columns=['generated_peptides'])
#     df.to_csv(output_path)
