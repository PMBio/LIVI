from datetime import timedelta
from pathlib import Path
from typing import Literal, Optional

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from typing_extensions import override


class LIVI_Checkpoint(ModelCheckpoint):
    """Overrides pytorch_lightning.callbacks.ModelCheckpoint to save model checkpoint only after
    VAE and adversary warm-up has been completed and individual effects have been introduced."""

    def __init__(
        self,
        dirpath: Optional[Path] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[Literal[True, False, "link"]] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: bool = True,
    ):
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode="min",
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end,
        )

    def _should_save_on_train_epoch_end(self, trainer: Trainer) -> bool:
        if self._save_on_train_epoch_end is not None:
            return self._save_on_train_epoch_end

    @override
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Save a checkpoint at the end of the training epoch."""
        save = trainer.current_epoch + 1 > pl_module.checkpointing_epoch
        if not save:
            print("Checkpointing epoch not reached yet. LIVI checkpointing disabled.")
        if (
            not self._should_skip_saving_checkpoint(trainer)
            and self._should_save_on_train_epoch_end(trainer)
            and save
        ):
            monitor_candidates = self._monitor_candidates(trainer)
            if (
                self._every_n_epochs >= 1
                and (trainer.current_epoch + 1) % self._every_n_epochs == 0
            ):
                self._save_topk_checkpoint(trainer, monitor_candidates)
            self._save_last_checkpoint(trainer, monitor_candidates)

    @override
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Save a checkpoint at the end of the validation stage."""
        save = trainer.current_epoch + 1 > pl_module.checkpointing_epoch
        if not save:
            print("Checkpointing epoch not reached yet. LIVI checkpointing disabled.")
        if (
            not self._should_skip_saving_checkpoint(trainer)
            and not self._should_save_on_train_epoch_end(trainer)
            and save
        ):
            monitor_candidates = self._monitor_candidates(trainer)
            if (
                self._every_n_epochs >= 1
                and (trainer.current_epoch + 1) % self._every_n_epochs == 0
            ):
                self._save_topk_checkpoint(trainer, monitor_candidates)
            self._save_last_checkpoint(trainer, monitor_candidates)
