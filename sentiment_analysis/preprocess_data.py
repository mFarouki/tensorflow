import random
import re
import string

import tensorflow as tf
from tensorflow.keras import layers

from utilities import display_info


def text_standardisation(input_data: tf.Tensor):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')


def vectorise_text_layer(max_features=10000, sequence_length=250):
    return layers.experimental.preprocessing.TextVectorization(
        standardize=text_standardisation, max_tokens=max_features, output_mode='int',
        output_sequence_length=sequence_length)


def apply_vectorisation(raw_train_ds: tf.data.Dataset):
    display_info('Learning vectors to represent ngrams present in data')
    train_text = raw_train_ds.map(lambda x, y: x)
    vectorised_layer = vectorise_text_layer()
    vectorised_layer.adapt(train_text)
    display_info(f'Built vocabulary of size: {len(vectorised_layer.get_vocabulary())}')
    return vectorised_layer


def vectorise_text(text: tf.Tensor, label: tf.Tensor, vectorised_layer: layers.Layer):
    text = tf.expand_dims(text, -1)
    return vectorised_layer(text), label


def vectorise_sample(raw_train_ds: tf.data.Dataset, vectorised_layer: layers.Layer):
    first_text_batch, label_batch = next(iter(raw_train_ds))
    review_number = random.randint(0, len(first_text_batch) - 1)
    display_info(f'Viewing vector representation of review {review_number} in the first batch')
    review, label = first_text_batch[review_number], label_batch[review_number]
    return review, label, vectorise_text(review, label, vectorised_layer)[0][0]


def view_sample_vectorisation(raw_train_ds: tf.data.Dataset, vectorised_layer: layers.Layer):
    review, label, vectorised_text = vectorise_sample(raw_train_ds, vectorised_layer)
    first_10_indices = vectorised_text[0:10]
    print(f'Review: {review}')
    print(f'Label: {raw_train_ds.class_names[label]}')
    print(f'''Vectorized review: 
{vectorised_text}''')
    for index in first_10_indices:
        print(f"{index} ---> {vectorised_layer.get_vocabulary()[index]}")


def preprocess_dataset(raw_ds: tf.data.Dataset, vectorised_layer: layers.Layer):
    vectorised_ds = raw_ds.map(lambda text, label: vectorise_text(text, label, vectorised_layer))
    return vectorised_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
