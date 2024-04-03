import pandas as pd
import torch
import esm
from tqdm import tqdm
import random


def generate_script(task_name,base_file_path,num_base_peps,num_peps_per_base,min_length,max_length,sample_variances_down,sample_variances_up,sample_variances_step,output_path):
    sample_variances = range(sample_variances_down,sample_variances_up, sample_variances_step)
    n = num_peps_per_base * num_base_peps # 最终生成数
    # 加载数据
    sample_df = pd.read_csv(base_file_path)
    # 加载编码用模型
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    do_have_GPU = torch.cuda.is_available()
    if do_have_GPU:
        model.cuda()
    # -- 生成算法部分 --
    # 先依照多肽设置的长短范围，将符合这个长度范围的能用作sample的数据从原库中筛出
    base_peptides = sample_df.loc[(sample_df['pep_len'] <= max_length) & (sample_df['pep_len'] >= min_length)].pep_seq.to_list()
    # 随机采样100个作为基底
    sampled_peptides = random.sample(base_peptides, num_base_peps)
    generated_peptides = []
    for pep in tqdm(sampled_peptides):
        target_seq = pep
        batch_labels, batch_strs, batch_tokens = batch_converter([("target_seq", target_seq)])
        if do_have_GPU:
            batch_tokens = batch_tokens.cuda()
        # batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        if do_have_GPU:
            token_representations = results["representations"][33].cpu()
        else:
            token_representations = results["representations"][33]
        del batch_tokens
        num_samples_per_base = int(int(n / num_base_peps) / len(sample_variances))
        for i in sample_variances:
            for j in range(num_samples_per_base):
                gen_pep = token_representations + torch.randn(
                    token_representations.shape) * i * token_representations.var()
                aa_toks = list("ARNDCEQGHILKMFPSTWYV")
                aa_idxs = [alphabet.get_idx(aa) for aa in aa_toks]
                if do_have_GPU:
                    aa_logits = model.lm_head(gen_pep.cuda())[:, :, aa_idxs]
                else:
                    aa_logits = model.lm_head(gen_pep)[:, :, aa_idxs]
                predictions = torch.argmax(aa_logits, dim=2).tolist()[0]
                generated_pep_seq = "".join([aa_toks[i] for i in predictions])
                generated_peptides.append(generated_pep_seq[1:-1])

    # -- 生成算法部分结束 --
    df = pd.DataFrame(generated_peptides, columns=['generated_peptides'])
    out_file_path = f'{output_path}/{task_name}.csv'
    df.to_csv(out_file_path)
    return out_file_path