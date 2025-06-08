from base.base_model import BaseModel
from keras.models import Sequential
from keras import layers


class ConvBloodCellModel(BaseModel):
    def __init__(self, config):
        super(ConvBloodCellModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = Sequential(
            [
                layers.Rescaling(1.0 / 255, input_shape=(256, 256, 3)),
                layers.Conv2D(128, (5, 5), strides=(3, 3), activation="relu"),
                layers.BatchNormalization(),
                layers.Conv2D(
                    256, (5, 5), strides=(2, 2), activation="relu", padding="same"
                ),
                layers.BatchNormalization(),
                layers.MaxPool2D(pool_size=(3, 3)),
                layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPool2D(),
                layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPool2D(),
                layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPool2D(),
                layers.Flatten(),
                layers.Dense(1024, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(1024, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(4, activation="softmax"),
            ]
        )

        self.model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
