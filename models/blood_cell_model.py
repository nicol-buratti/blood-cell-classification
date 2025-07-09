from base.base_model import BaseModel
from keras.models import Sequential
from keras import layers, regularizers


class ConvBloodCellModel(BaseModel):
    def __init__(self, config):
        super(ConvBloodCellModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        l2_reg = regularizers.l2(0.005)
        self.model = Sequential(
            [
                # Input and Rescaling
                layers.Input(shape=(256, 256, 3)),
                layers.Rescaling(1.0 / 255),

                # Data Augmentation Layers
                layers.RandomFlip("horizontal_and_vertical"),
                layers.RandomRotation(0.15), # Rotate by up to 15 degrees
                layers.RandomZoom(0.2),      # Zoom in/out by up to 20%
                layers.RandomContrast(0.2),  # Adjust contrast by up to 20%
                layers.RandomTranslation(height_factor=0.1, width_factor=0.1), # Shift images

                # First Convolutional Block
                layers.Conv2D(32, (5, 5), strides=(2, 2), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPool2D(pool_size=(2, 2)), 
                layers.Dropout(0.25),

                # Second Convolutional Block
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPool2D(pool_size=(2, 2)),
                layers.Dropout(0.3),

                # Third Convolutional Block
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPool2D(pool_size=(2, 2)),
                layers.Dropout(0.35),

                # Fourth Convolutional Block
                layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPool2D(pool_size=(2, 2)),
                layers.Dropout(0.4),

                 # Classifier Head
                layers.GlobalAveragePooling2D(), # Efficiently flattens feature maps
                layers.Dense(512, activation="relu", kernel_regularizer=l2_reg),
                layers.BatchNormalization(), 
                layers.Dropout(0.5), 

                layers.Dense(256, activation="relu", kernel_regularizer=l2_reg),
                layers.BatchNormalization(),
                layers.Dropout(0.4),

                layers.Dense(4, activation="softmax"), # Output layer for 4 classes
            ]
        )

        self.model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        print(self.model.summary())