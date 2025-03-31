from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from transformer import SmallGPT2
import torch

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')

    checkpoint_callback = ModelCheckpoint(
        dirpath="/content/drive/MyDrive/bachgencheckpoints",
        filename="bach_gen_{epoch:02d}_{loss:.2f}",
        every_n_train_steps=100,
        save_top_k=1,
    )
    model = SmallGPT2()
    trainer = Trainer(accelerator="gpu", callbacks=[checkpoint_callback], max_epochs=3)
    trainer.fit(model)
