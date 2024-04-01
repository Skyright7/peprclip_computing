# 使用训练好的clip模型
# 输入：单行文本(想要计算的受体蛋白质氨基酸序列)，候选多肽csv的地址。
# 输出：csv文件(内容是依照用户意愿个数的筛选结果)

import pandas as pd
from tqdm import tqdm
import torch
import esm
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

# import argparse
#
# parser = argparse.ArgumentParser(description='Clip model of computing.')
# parser.add_argument('-generated_peptides_path',type=str, required=True)
# parser.add_argument('-peps_per_target',type=int, required=True)
# parser.add_argument('-target_seq',type=str, required=True)
# parser.add_argument('-target_name',type=str, required=True)
# parser.add_argument('-output_base_path',type=str, required=True)
# args = parser.parse_args()
#
# generated_peptides_path = args.generated_peptides_path
# peps_per_target = args.peps_per_target # 每个目标蛋白最终筛选出几个配体
# target_seq = args.target_seq
# target_name = args.target_name
# output_base_path = args.output_base_path

# 模型源码，在使用部分仅用于加载模型
class MiniCLIP(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        ##protein encoding: 2 layers, latent space of size 320?

        self.prot_embedder = nn.Sequential(
          nn.Linear(1280, 640),
          nn.ReLU(),
          nn.Linear(640, 320),
        )

        ##peptide encoding: start with 2 layers, may want to add in a decoder later
        self.pep_embedder = nn.Sequential(
          nn.Linear(1280, 640),
          nn.ReLU(),
          nn.Linear(640, 320),
        )

    def forward(self, pep_input, prot_input):
        ##get peptide and protein embeddings, dot together
        pep_embedding = F.normalize(self.pep_embedder(pep_input))
        prot_embedding = F.normalize(self.prot_embedder(prot_input))

        logits = torch.matmul(pep_embedding, prot_embedding.T) ##may need to transpose something here

        return logits

    def training_step(self, batch, batch_idx):

        logits = self(
            batch['peptide_input'],
            batch['protein_input'],
        )

        batch_size = batch['peptide_input'].shape[0]
        labels = torch.arange(batch_size).to(self.device) ##NOTE: to(self.device) is important here
                 ##this gives us the diagonal clip loss structure

        # loss of predicting partner using peptide
        partner_prediction_loss = F.cross_entropy(logits, labels)

        # loss of predicting peptide using partner
        peptide_prediction_loss = F.cross_entropy(logits.T, labels)

        loss = (partner_prediction_loss + peptide_prediction_loss) / 2

        self.log("train_loss", loss, sync_dist=True, batch_size=logits.shape[0])
        self.log("train_partner_prediction_loss", partner_prediction_loss, sync_dist=True, prog_bar=False, batch_size=logits.shape[0])
        self.log("train_peptide_prediction_loss", peptide_prediction_loss, sync_dist=True, prog_bar=False, batch_size=logits.shape[0])

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0 or dataloader_idx == 2:
          if dataloader_idx == 0:
            prefix = "noisy"
          else:
            prefix = "strict"

          # Predict on random batches of training batch size
          logits = self(
              batch['peptide_input'],
              batch['protein_input'],
          )

          batch_size = batch['peptide_input'].shape[0]
          labels = torch.arange(batch_size).to(self.device) ##NOTE: to(self.device) is important here
          ##this gives us the diagonal clip loss structure

          # loss of predicting partner using peptide
          partner_prediction_loss = F.cross_entropy(logits, labels)

          # loss of predicting peptide using partner
          peptide_prediction_loss = F.cross_entropy(logits.T, labels)

          loss = (partner_prediction_loss + peptide_prediction_loss) / 2


          # prediction of peptides for each partner
          peptide_predictions = logits.argmax(dim=0)
          # prediction of partners for each peptide
          partner_predictions = logits.argmax(dim=1)

          peptide_ranks = logits.argsort(dim=0).diag() + 1
          peptide_mrr = (peptide_ranks).float().pow(-1).mean()

          partner_ranks = logits.argsort(dim=1).diag() + 1
          partner_mrr = (partner_ranks).float().pow(-1).mean()

          partner_accuracy = partner_predictions.eq(labels).float().mean()
          peptide_accuracy = peptide_predictions.eq(labels).float().mean()

          k = int(logits.shape[0] / 10)
          peptide_topk_accuracy = torch.any((logits.topk(k, dim=0).indices - labels.reshape(1, -1)) == 0, dim=0).sum() / logits.shape[0]
          partner_topk_accuracy = torch.any((logits.topk(k, dim=1).indices - labels.reshape(-1, 1)) == 0, dim=1).sum() / logits.shape[0]


          self.log(f"{prefix}_val_loss", loss, sync_dist=True, prog_bar=False, batch_size=logits.shape[0], add_dataloader_idx=False)
          self.log(f"{prefix}_val_perplexity", torch.exp(loss), sync_dist=False, prog_bar=True, batch_size=logits.shape[0], add_dataloader_idx=False)
          self.log(f"{prefix}_val_partner_prediction_loss", partner_prediction_loss, sync_dist=True, prog_bar=False, batch_size=logits.shape[0], add_dataloader_idx=False)
          self.log(f"{prefix}_val_peptide_prediction_loss", peptide_prediction_loss, sync_dist=True, prog_bar=False, batch_size=logits.shape[0], add_dataloader_idx=False)
          self.log(f"{prefix}_val_partner_perplexity", torch.exp(partner_prediction_loss), sync_dist=True, prog_bar=False, batch_size=logits.shape[0], add_dataloader_idx=False)
          self.log(f"{prefix}_val_peptide_perplexity", torch.exp(peptide_prediction_loss), sync_dist=True, prog_bar=True, batch_size=logits.shape[0], add_dataloader_idx=False)
          self.log(f"{prefix}_val_partner_accuracy", partner_accuracy, sync_dist=True, prog_bar=False, batch_size=logits.shape[0], add_dataloader_idx=False)
          self.log(f"{prefix}_val_peptide_accuracy", peptide_accuracy, sync_dist=True, prog_bar=False, batch_size=logits.shape[0], add_dataloader_idx=False)
          self.log(f"{prefix}_val_partner_top10p", partner_topk_accuracy, sync_dist=True, prog_bar=False, batch_size=logits.shape[0], add_dataloader_idx=False)
          self.log(f"{prefix}_val_peptide_top10p", peptide_topk_accuracy, sync_dist=True, prog_bar=True, batch_size=logits.shape[0], add_dataloader_idx=False)
          self.log(f"{prefix}_val_peptide_mrr", peptide_mrr, sync_dist=True, prog_bar=False, batch_size=logits.shape[0], add_dataloader_idx=False)
          self.log(f"{prefix}_val_partner_mrr", partner_mrr, sync_dist=True, prog_bar=False, batch_size=logits.shape[0], add_dataloader_idx=False)

        else:
          if dataloader_idx == 1:
            prefix = "noisy"
          else:
            prefix = "strict"

          # Given a protein, predict the correct peptide out of 2
          logits = self(
              batch['peptide_input'],
              batch['protein_input'],
          )

          batch_size = batch['peptide_input'].shape[0]
          labels = torch.arange(batch_size).to(self.device) ##NOTE: to(self.device) is important here
          ##this gives us the diagonal clip loss structure


          binary_cross_entropy = F.cross_entropy(logits.T, labels)

          binary_predictions = logits.argmax(dim=0)
          binary_accuracy = binary_predictions.eq(labels).float().mean()

          self.log(f"{prefix}_binary_loss", binary_cross_entropy, sync_dist=True, prog_bar=False, batch_size=2, add_dataloader_idx=False)
          self.log(f"{prefix}_binary_accuracy", binary_accuracy, sync_dist=False, prog_bar=True, batch_size=2, add_dataloader_idx=False)


    def test_step(self, batch, batch_idx, dataloader_idx=0):

        if dataloader_idx == 0 or dataloader_idx == 2:
          if dataloader_idx == 0:
            prefix = "noisy"
          else:
            prefix = "strict"

          # Predict on random batches of training batch size
          logits = self(
              batch['peptide_input'],
              batch['protein_input'],
          )

          batch_size = batch['peptide_input'].shape[0]
          labels = torch.arange(batch_size).to(self.device) ##NOTE: to(self.device) is important here
          ##this gives us the diagonal clip loss structure

          # loss of predicting partner using peptide
          partner_prediction_loss = F.cross_entropy(logits, labels)

          # loss of predicting peptide using partner
          peptide_prediction_loss = F.cross_entropy(logits.T, labels)

          loss = (partner_prediction_loss + peptide_prediction_loss) / 2


          # prediction of peptides for each partner
          peptide_predictions = logits.argmax(dim=0)
          # prediction of partners for each peptide
          partner_predictions = logits.argmax(dim=1)

          peptide_ranks = logits.argsort(dim=0).diag() + 1
          peptide_mrr = (peptide_ranks).float().pow(-1).mean()

          partner_ranks = logits.argsort(dim=1).diag() + 1
          partner_mrr = (partner_ranks).float().pow(-1).mean()

          partner_accuracy = partner_predictions.eq(labels).float().mean()
          peptide_accuracy = peptide_predictions.eq(labels).float().mean()

          k = int(logits.shape[0] / 10)
          peptide_topk_accuracy = torch.any((logits.topk(k, dim=0).indices - labels.reshape(1, -1)) == 0, dim=0).sum() / logits.shape[0]
          partner_topk_accuracy = torch.any((logits.topk(k, dim=1).indices - labels.reshape(-1, 1)) == 0, dim=1).sum() / logits.shape[0]


          self.log(f"{prefix}_test_loss", loss, sync_dist=True, prog_bar=False, batch_size=logits.shape[0], add_dataloader_idx=False)
          self.log(f"{prefix}_test_perplexity", torch.exp(loss), sync_dist=False, prog_bar=True, batch_size=logits.shape[0], add_dataloader_idx=False)
          self.log(f"{prefix}_test_partner_prediction_loss", partner_prediction_loss, sync_dist=True, prog_bar=False, batch_size=logits.shape[0], add_dataloader_idx=False)
          self.log(f"{prefix}_test_peptide_prediction_loss", peptide_prediction_loss, sync_dist=True, prog_bar=False, batch_size=logits.shape[0], add_dataloader_idx=False)
          self.log(f"{prefix}_test_partner_perplexity", torch.exp(partner_prediction_loss), sync_dist=True, prog_bar=False, batch_size=logits.shape[0], add_dataloader_idx=False)
          self.log(f"{prefix}_test_peptide_perplexity", torch.exp(peptide_prediction_loss), sync_dist=True, prog_bar=True, batch_size=logits.shape[0], add_dataloader_idx=False)
          self.log(f"{prefix}_test_partner_accuracy", partner_accuracy, sync_dist=True, prog_bar=False, batch_size=logits.shape[0], add_dataloader_idx=False)
          self.log(f"{prefix}_test_peptide_accuracy", peptide_accuracy, sync_dist=True, prog_bar=False, batch_size=logits.shape[0], add_dataloader_idx=False)
          self.log(f"{prefix}_test_partner_top10p", partner_topk_accuracy, sync_dist=True, prog_bar=False, batch_size=logits.shape[0], add_dataloader_idx=False)
          self.log(f"{prefix}_test_peptide_top10p", peptide_topk_accuracy, sync_dist=True, prog_bar=True, batch_size=logits.shape[0], add_dataloader_idx=False)
          self.log(f"{prefix}_test_peptide_mrr", peptide_mrr, sync_dist=True, prog_bar=False, batch_size=logits.shape[0], add_dataloader_idx=False)
          self.log(f"{prefix}_test_partner_mrr", partner_mrr, sync_dist=True, prog_bar=False, batch_size=logits.shape[0], add_dataloader_idx=False)

        else:
          if dataloader_idx == 1:
            prefix = "noisy"
          else:
            prefix = "strict"

          # Given a protein, predict the correct peptide out of 2
          logits = self(
              batch['peptide_input'],
              batch['protein_input'],
          )

          batch_size = batch['peptide_input'].shape[0]
          labels = torch.arange(batch_size).to(self.device) ##NOTE: to(self.device) is important here
          ##this gives us the diagonal clip loss structure


          binary_cross_entropy = F.cross_entropy(logits.T, labels)

          binary_predictions = logits.argmax(dim=0)
          binary_accuracy = binary_predictions.eq(labels).float().mean()

          self.log(f"{prefix}_test_binary_loss", binary_cross_entropy, sync_dist=True, prog_bar=False, batch_size=2, add_dataloader_idx=False)
          self.log(f"{prefix}_test_binary_accuracy", binary_accuracy, sync_dist=False, prog_bar=True, batch_size=2, add_dataloader_idx=False)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)



# generated_peptides_path = './data/generate_demo_data/output/generated_peptides.csv'
# peps_per_target = 10 # 每个目标蛋白最终筛选出几个配体
# target_seq = 'MGVPRPQPWALGLLLFLLPGSLGAESHLSLLYHLTAVSSPAPGTPAFWVSGWLGPQQYLSYNSLRGEAEPCGAWVWENQVSWYWEKETTDLRIKEKLFLEAFKALGGKGPYTLQGLLGCELGPDNTSVPTAKFALNGEEFMNFDLKQGTWGGDWPEALAISQRWQQQDKAANKELTFLLFSCPHRLREHLERGRGNLEWKEPPSMRLKARPSSPGFSVLTCSAFSFYPPELQLRFLRNGLAAGTGQGDFGPNSDGSFHASSSLTVKSGDEHHYCCIVQHAGLAQPLRVELESPAKSSVLVVGIVIGVLLLTAAAVGGALLWRRMRSGLPAPWISLRGDDTGVLLPTPGEAQDADLKDVNVIPATA'
# target_name = 'FcRn'

def peptide_encoding_preparation(generated_peptides_path,model, alphabet,batch_converter,do_have_GPU):
    de_novo_peptides_df = pd.read_csv(generated_peptides_path)
    de_novo_peptides = de_novo_peptides_df['generated_peptides'].tolist()
    candidate_peptide_dict = {}
    for candidate_peptide in tqdm(de_novo_peptides):
        batch_labels, batch_strs, batch_tokens = batch_converter([("candidate_peptide", candidate_peptide)])
        if do_have_GPU:
            batch_tokens = batch_tokens.cuda()
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)

        if do_have_GPU:
            token_representations = results["representations"][33].cpu()
        else:
            token_representations = results["representations"][33]
        del batch_tokens

        sequence_representations = []
        for j, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[j, 1: tokens_len - 1].mean(0))

        candidate_peptide_embedding = sequence_representations[0]
        candidate_peptide_dict.update({candidate_peptide: candidate_peptide_embedding})
    return candidate_peptide_dict

def do_clip(target_seq,target_name, generated_peptides_path,peps_per_target,output_base_path,task_name,model_weight_path):
    # 加载编码器模型
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    do_have_GPU = torch.cuda.is_available()
    do_have_GPU = False
    if do_have_GPU:
        model.cuda()
    candidate_peptide_dict = peptide_encoding_preparation(generated_peptides_path,model, alphabet,batch_converter,do_have_GPU)
    all_candidate_peptides = list(candidate_peptide_dict.keys())
    # 加载模型参数
    miniclip = MiniCLIP.load_from_checkpoint(model_weight_path, lr=0.003)
    targets = {target_name:target_seq}
    output_dict = {}
    for name, target_seq in tqdm(targets.items()):
        ##feed sequence it into ESM
        batch_labels, batch_strs, batch_tokens = batch_converter([("target_seq", target_seq)])
        if do_have_GPU:
            batch_tokens = batch_tokens.cuda()

        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)

        if do_have_GPU:
            token_representations = results["representations"][33].cpu()
        else:
            token_representations = results["representations"][33]

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
    file_out_path = f'{output_base_path}/{target_name}_{task_name}.csv'
    peptides_df.to_csv(file_out_path)
    return file_out_path

