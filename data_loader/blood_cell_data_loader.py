from pathlib import Path
from base.base_data_loader import BaseDataLoader
import tensorflow as tf


class BloodCellDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(BloodCellDataLoader, self).__init__(config)

        train_path = Path(config.data_loader.images_path) / "TRAIN"
        test_path = Path(config.data_loader.images_path) / "TEST"

        self.train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            train_path, image_size=(224, 224), batch_size=32, label_mode="categorical"
        )

        self.test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            test_path,
            image_size=(224, 224),
            batch_size=32,
            label_mode="categorical",
        )

    def get_train_data(self):
        return self.train_dataset

    def get_test_data(self):
        return self.test_dataset
