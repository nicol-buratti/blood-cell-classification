from pathlib import Path
from base.base_data_loader import BaseDataLoader
import tensorflow as tf
from keras import layers


class BloodCellDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(BloodCellDataLoader, self).__init__(config)

        train_path = Path(config.data_loader.images_path) / "TRAIN"
        test_path = Path(config.data_loader.images_path) / "TEST"

        self.data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ])

        self.train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            train_path,
            batch_size=config.data_loader.batch_size,
            label_mode="categorical",
            image_size=(256, 256),
            shuffle=True,
            seed=getattr(config.exp, 'seed', 42)
        )

        self.test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            test_path,
            batch_size=config.data_loader.batch_size,
            label_mode="categorical",
            image_size=(256, 256),
            shuffle=False
        )

        self.class_names = self.train_dataset.class_names
        print(f"Found {len(self.class_names)} classes: {self.class_names}")

        AUTOTUNE = tf.data.AUTOTUNE

        self.train_dataset = self.train_dataset.map(
            lambda x, y: (self.data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        ).prefetch(buffer_size=AUTOTUNE)

        self.test_dataset = self.test_dataset.prefetch(buffer_size=AUTOTUNE)

    def get_train_data(self):
        return self.train_dataset

    def get_test_data(self):
        return self.test_dataset

    def get_class_names(self):
        return self.class_names