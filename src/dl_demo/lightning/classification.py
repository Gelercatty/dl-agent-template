"""Task loss, optimizer, metrics and task-specific checkpoint state."""

import pytorch_lightning as L
import torch
from torch.nn import functional as F

from workflow.common import require


class ClassificationModule(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model, self.config = model, config
        self.initial_loss = None
        self.resumed_from_step = 0
        self.validation_steps = []

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self.model(x), y)
        require(bool(torch.isfinite(loss)), "Nonfinite training loss")
        if self.initial_loss is None:
            self.initial_loss = float(loss.detach())
        self.log("train/loss", loss, on_step=True, on_epoch=False, batch_size=len(y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        self.log("val/loss", F.cross_entropy(logits, y), batch_size=len(y))
        self.log("val/accuracy", (logits.argmax(-1) == y).float().mean(), batch_size=len(y))

    def on_validation_epoch_end(self):
        if not self.validation_steps or self.validation_steps[-1] != self.global_step:
            self.validation_steps.append(self.global_step)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config["optimizer"]["learning_rate"])

    def on_save_checkpoint(self, checkpoint):
        checkpoint["task_state"] = {
            "initial_loss": self.initial_loss,
            "validation_steps": self.validation_steps,
            "torch_rng": torch.get_rng_state(),
        }

    def on_load_checkpoint(self, checkpoint):
        state = checkpoint["task_state"]
        self.initial_loss = state["initial_loss"]
        self.validation_steps = list(state["validation_steps"])
        self.resumed_from_step = checkpoint["global_step"]
        torch.set_rng_state(state["torch_rng"])
