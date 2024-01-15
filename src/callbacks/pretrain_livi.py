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
 
            
class LIVIPretrain(Callback):
    def __init__(self, pretrain_epochs=10, pretrain_wo_GxC=True):
        super().__init__()
        self._pretrain_epochs = pretrain_epochs
        self._genetics_epochs = pretrain_epochs+10 if pretrain_wo_GxC else pretrain_epochs

    def on_train_start(self, trainer, pl_module):
        pl_module.set_pretrain_mode(True)
        pl_module.set_pretrain_G_mode(True)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == self._pretrain_epochs:
            print("VAE pretraining completed")
            pl_module.set_pretrain_mode(False)
        if trainer.current_epoch == self._genetics_epochs:
            print("Start learning GxC effects")
            pl_module.set_pretrain_G_mode(False)
