import os
import tensorflow as tf
from . import config
from .data_loader import load_datasets
from .model import CrackDetectionModel

def train():
    
    # Prepare data
    print("Loading datasets...")
    train_ds, val_ds, _, class_names = load_datasets()
    
    # Get number of classes
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")

    # Build model
    print("Building model...")
    model_builder = CrackDetectionModel(num_classes=num_classes)
    model = model_builder.build()

    # Compile model
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY
    )
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # Ensure output directory exists
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(config.OUTPUT_DIR, "best_model.keras")

    # Define callbacks
    callbacks = [
        # Saves the model every time validation accuracy improves
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Stops training if validation accuracy doesn't improve for 10 epochs
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Lowers learning rate if progress stalls after 5 epochs
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.1,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Run training
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=callbacks
    )

    return history

if __name__ == "__main__":
    train()