import tensorflow as tf
from tensorflow.keras import layers, models
from .data_loader import get_data_augmentation
from tensorflow.keras.applications.inception_v3 import InceptionV3
from . import config

class CrackDetectionModel:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.image_shape = config.IMAGE_SHAPE + (3,) # Add 3 RGB channels

    def build(self):
        """Constructs the model architecture."""
        
        # Base pre-trained model
        base_model = InceptionV3(
            input_shape=self.image_shape,
            include_top=False,
            weights='imagenet'
        )

        # Freeze the base model
        base_model.trainable = False

        # Building the model
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.image_shape),
            
            # Augmentation layer
            get_data_augmentation(),

            # Preprocessing layer
            layers.Rescaling(1./255),
            
            # InceptionV3 feature extractor
            base_model,
            
            # Custom layers
            layers.Flatten(),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.BatchNormalization(),
            
            # Final output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])

        return model