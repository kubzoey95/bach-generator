from transformers import GPT2Config, GPT2LMHeadModel
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_model import MusicDataset


class SmallGPT2(pl.LightningModule):
    def __init__(self, seq_len=256, vocab_size=39):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        n_embd = 256
        custom_config = GPT2Config(
            vocab_size=self.vocab_size,
            n_positions=self.seq_len,
            n_ctx=self.seq_len,
            n_embd=n_embd,
            n_layer=6,
            n_head=8,
            max_length=self.seq_len,
        )
        self.lm_head = torch.nn.Linear(n_embd, self.vocab_size)
        self.model = GPT2LMHeadModel(custom_config).base_model

        self.loss = torch.nn.CrossEntropyLoss()
        self.dataset = MusicDataset("baroque_processed", context_length=self.seq_len)

    def forward(self, *args, **kwargs):
        backbone_out = self.model.forward(*args, **kwargs).last_hidden_state
        logits = self.lm_head(backbone_out).permute(0, 2, 1)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_ = torch.concat([x[:, :-1], y.unsqueeze(-1)], dim=-1)

        loss = F.cross_entropy(self.forward(x), y_)
        self.log("loss", loss.item(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-3,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,
            anneal_strategy='cos',
            final_div_factor=1e4,
        )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def train_dataloader(self):
        return DataLoader(self.dataset, 46, shuffle=True, num_workers=46)
