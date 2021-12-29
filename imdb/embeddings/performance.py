import tensorflow as tf


def presets(dataset: tf.data.Dataset):
    return dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
