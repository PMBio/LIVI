from pytorch_lightning.callbacks import Callback


class LIVIPretrainVAE(Callback):
    def __init__(self, pretrain_epochs=10):
        super().__init__()
        self._pretrain_epochs = pretrain_epochs

    def on_train_start(self, trainer, pl_module):
        pl_module.set_pretrain_mode(True)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == self._pretrain_epochs:
            print("VAE pretraining completed")
            pl_module.set_pretrain_mode(False)
