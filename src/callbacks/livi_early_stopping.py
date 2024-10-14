from datetime import timedelta
from pathlib import Path
from typing import Optional

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from typing_extensions import override


class LIVI_EarlyStopping(EarlyStopping):
    """Overrides pytorch_lightning.callbacks.EarlyStopping to activate early stopping only after
    VAE and adversary warm-up has been completed and individual effects have been introduced."""

    def __init__(
        self,
        monitor: str,
        min_delta: float = 0.0,
        patience: int = 10,
        verbose: bool = False,
        strict: bool = True,
        stopping_threshold: Optional[float] = None,
        divergence_threshold: Optional[float] = None,
        check_on_train_epoch_end: Optional[bool] = True,
        log_rank_zero_only: bool = False,
    ):
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode="min",
            strict=strict,
            check_finite=True,
            stopping_threshold=stopping_threshold,
            divergence_threshold=divergence_threshold,
            check_on_train_epoch_end=check_on_train_epoch_end,
            log_rank_zero_only=log_rank_zero_only,
        )

    @override
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Save a checkpoint at the end of the training epoch."""
        if (
            trainer.current_epoch + 1 < pl_module.checkpointing_epoch
            or trainer.current_epoch + 1 < trainer.min_epochs
        ):
            self.wait_count = 0
            print("Early stopping epoch not reached yet. LIVI early stopping disabled.")
            return
        else:
            self.wait_count = self.wait_count
            if not self._check_on_train_epoch_end or self._should_skip_check(trainer):
                return
            print(f"LIVI early stopping enabled. Epoch wait count: {self.wait_count}")
            self._run_early_stopping_check(trainer)

    @override
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Save a checkpoint at the end of the validation stage."""
        if (
            trainer.current_epoch + 1 < pl_module.checkpointing_epoch
            or trainer.current_epoch + 1 < trainer.min_epochs
        ):
            self.wait_count = 0
            print("Early stopping epoch not reached yet. LIVI early stopping disabled.")
            return
        else:
            self.wait_count = self.wait_count
            if self._check_on_train_epoch_end or self._should_skip_check(trainer):
                return
            print(f"LIVI early stopping enabled. Epoch wait count: {self.wait_count}")
            self._run_early_stopping_check(trainer)
