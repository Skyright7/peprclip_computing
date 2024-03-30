# 模型源码，在使用部分仅用于加载模型

import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

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

