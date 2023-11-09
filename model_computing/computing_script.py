import pandas as pd
from tqdm import tqdm
import pickle
import torch
import esm
from model_computing.MiniClip_model import MiniCLIP

def out_computing_script(target_name,target_seq, pre_gen_sample_url, checkpoint_url,num_per_target):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    miniclip = MiniCLIP.load_from_checkpoint(checkpoint_url, lr=0.003)
    peps_per_target = num_per_target
    targets = {target_name: target_seq}
    with open(pre_gen_sample_url, "rb") as f:
        candidate_peptide_dict = pickle.load(f)
    all_candidate_peptides = list(candidate_peptide_dict.keys())
    ##now, for each target, get the ESM embedding and clip it against the candidate peptide dictionary
    output_dict = {}
    for name, target_seq in tqdm(targets.items()):
        ##feed sequence it into ESM
        batch_labels, batch_strs, batch_tokens = batch_converter([("target_seq", target_seq)])
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)

        token_representations = results["representations"][33].cpu()
        del batch_tokens

        sequence_representations = []
        for j, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[j, 1: tokens_len - 1].mean(0))

        target_prot_embedding = sequence_representations[0]

        peptide_scores = []
        seq_to_score_dict = {}

        for candidate_peptide, candidate_peptide_embedding in candidate_peptide_dict.items():
            score = miniclip.forward(candidate_peptide_embedding.unsqueeze(0), target_prot_embedding.unsqueeze(0))
            peptide_scores.append(score)
            seq_to_score_dict.update({candidate_peptide: score})

        topk_idxs = list(torch.concat(peptide_scores).argsort(dim=0, descending=True)[:peps_per_target])
        topk_peptides = [all_candidate_peptides[topk_idxs[i]] for i in range(len(topk_idxs))]
        topk_scores = [float(seq_to_score_dict[peptide]) for peptide in topk_peptides]

        output_dict.update({name: (topk_peptides, topk_scores)})
    peptides_df = pd.DataFrame(
        sum([list(zip([key + '_' + str(i) for i in range(peps_per_target)], output_dict[key][0], output_dict[key][1])) \
             for key in output_dict.keys()], []), columns=['name', 'sequence', 'clip_score'])
    peptides_df = peptides_df.sort_values('name')
    out_json = peptides_df.to_json(orient="index")
    return out_json

# if __name__ == '__main__':
#     #test
#     target_name = 'FcRn'
#     target_seq = 'MGVPRPQPWALGLLLFLLPGSLGAESHLSLLYHLTAVSSPAPGTPAFWVSGWLGPQQYLSYNSLRGEAEPCGAWVWENQVSWYWEKETTDLRIKEKLFLEAFKALGGKGPYTLQGLLGCELGPDNTSVPTAKFALNGEEFMNFDLKQGTWGGDWPEALAISQRWQQQDKAANKELTFLLFSCPHRLREHLERGRGNLEWKEPPSMRLKARPSSPGFSVLTCSAFSFYPPELQLRFLRNGLAAGTGQGDFGPNSDGSFHASSSLTVKSGDEHHYCCIVQHAGLAQPLRVELESPAKSSVLVVGIVIGVLLLTAAAVGGALLWRRMRSGLPAPWISLRGDDTGVLLPTPGEAQDADLKDVNVIPATA'
#     pre_gen_sample_url = './100k_denovo_for_FcRn.pkl'
#     checkpoint_url = './pepprclip_2023-10-12.ckpt'
#     num_per_target = 10
#     output = out_computing_script(target_name,target_seq,pre_gen_sample_url,checkpoint_url,num_per_target)
#     print(output)