import tensorflow as tf
from tensorflow.keras import layers
from . import config

def get_data_augmentation():
    """Defines the data augmentation pipeline."""
    
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    ])

def load_datasets():
    """Loads train, validation, and test datasets."""
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.DATA_DIR,
        validation_split=config.VAL_RATIO + config.TEST_RATIO,
        subset="training",
        seed=1337,
        image_size=config.IMAGE_SHAPE,
        batch_size=config.BATCH_SIZE,
    )

    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        config.DATA_DIR,
        validation_split=config.VAL_RATIO + config.TEST_RATIO,
        subset="validation",
        seed=1337,
        image_size=config.IMAGE_SHAPE,
        batch_size=config.BATCH_SIZE,
    )

    # Split the val_test_ds into separate val and test datasets
    total_val_test_batches = tf.data.experimental.cardinality(val_test_ds).numpy()
    val_batches = int(total_val_test_batches * (config.VAL_RATIO /
    (config.VAL_RATIO + config.TEST_RATIO)))
    
    val_ds = val_test_ds.take(val_batches)
    test_ds = val_test_ds.skip(val_batches)

    # Capture class names
    class_names = train_ds.class_names

    # Prefetching and caching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names