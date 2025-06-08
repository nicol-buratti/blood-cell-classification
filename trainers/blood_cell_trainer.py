from base.base_trainer import BaseTrain
import os
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping


class ConvBloodCellTrainer(BaseTrain):
    def __init__(self, model, data, config):
        super(ConvBloodCellTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(
                self.config.callbacks.checkpoint_dir,
                "%s-{epoch:02d}-{val_loss:.2f}.weights.h5" % self.config.exp.name,
            ),
            monitor=self.config.callbacks.checkpoint_monitor,
            mode=self.config.callbacks.checkpoint_mode,
            save_best_only=self.config.callbacks.checkpoint_save_best_only,
            save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
            verbose=self.config.callbacks.checkpoint_verbose,
        )

        tensorboard_callback = TensorBoard(
            log_dir=self.config.callbacks.tensorboard_log_dir,
            write_graph=self.config.callbacks.tensorboard_write_graph,
        )

        early_stopping = EarlyStopping(
            monitor=self.config.callbacks.earlystopping_monitor,
            patience=self.config.callbacks.earlystopping_patience,
        )

        self.callbacks = [checkpoint_callback, tensorboard_callback, early_stopping]

    def train(self):
        history = self.model.fit(
            self.data[0],
            validation_data = self.data[1],
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            batch_size=self.config.trainer.batch_size,
            callbacks=self.callbacks,
        )
        self.loss.extend(history.history["loss"])
        self.acc.extend(history.history["accuracy"])
        self.val_loss.extend(history.history["val_loss"])
        self.val_acc.extend(history.history["val_accuracy"])
