import sys

import tensorflow as tf

sys.path.append('../')
from utilities.utility_functions import display_info


def define_label(raw_train_ds: tf.data.Dataset, label: int):
    if (label >= len(raw_train_ds.class_names)) or (label < 0):
        raise ValueError(f'Label {label} not recognised in dataset class names')
    return raw_train_ds.class_names[label]


def build_raw_datasets(directory: str, batch_size: int, seed: int, validation_split=0.2):
    display_info(f'Reading train, validation, and test sets from {directory}/train')
    train_dataset = tf.keras.preprocessing.text_dataset_from_directory(
        f'{directory}/train', batch_size=batch_size, validation_split=validation_split, subset='training', seed=seed)
    validation_dataset = tf.keras.preprocessing.text_dataset_from_directory(
        f'{directory}/train', batch_size=batch_size, validation_split=validation_split, subset='validation', seed=seed)
    test_dataset = tf.keras.preprocessing.text_dataset_from_directory(f'{directory}/test', batch_size=batch_size)
    return train_dataset, validation_dataset, test_dataset


def view_dataset(raw_ds: tf.data.Dataset, n_reviews=3):
    n_batches = tf.data.experimental.cardinality(raw_ds)
    display_info(f'Viewing the first {n_reviews} reviews in the first batch (of {n_batches} total batches)')
    for text_batch, label_batch in raw_ds.take(1):
        batch_size = text_batch.shape[0]
        if n_reviews > batch_size:
            raise ValueError(f'Cannot show {n_reviews} reviews, because a batch only contains {batch_size} reviews')
        for i in range(n_reviews):
            review, label = text_batch.numpy()[i], label_batch.numpy()[i]
            print(f'Review: {review}')
            print(f'Label: {label}, indicating a {define_label(raw_ds, label)} review')

