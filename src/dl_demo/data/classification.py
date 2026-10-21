"""Seeded train/validation data; no model or optimization code."""

import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset


class ClassificationData(L.LightningDataModule):
    def __init__(self, config, seed, train_batches):
        super().__init__()
        self.config, self.seed = config, seed
        self.train_batches = train_batches

    def setup(self, stage=None):
        generator = torch.Generator().manual_seed(self.seed + 1000)

        def dataset():
            y = torch.randint(0, 2, (self.config["samples"],), generator=generator)
            x = torch.randn(self.config["samples"], 2, generator=generator) * 0.35
            x += (y.float() * 2 - 1).unsqueeze(1)
            return TensorDataset(x, y)

        self.train, self.val = dataset(), dataset()

    def _loader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=False,
            generator=torch.Generator().manual_seed(self.seed + 2000),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_sampler=[list(range(self.config["samples"]))] * self.train_batches,
            num_workers=0,
            generator=torch.Generator().manual_seed(self.seed + 2000),
        )

    def val_dataloader(self):
        return self._loader(self.val)
