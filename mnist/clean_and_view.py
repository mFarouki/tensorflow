import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def version_check():
    print(f'Running TensorFlow version {tf.__version__}')


def load_data(dataset):
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    return x_train, y_train, x_test, y_test


def rescale_max_1(np_array):
    if np.issubdtype(np_array.dtype, np.number):
        return np_array / np.amax(np_array)
    else:
        raise ValueError("Function can only rescale numeric values")


def describe_categorical_dataset(x_train, y_train, y_test):
    n_training_points = y_train.shape[0]
    n_test_points = y_test.shape[0]
    input_shape = x_train.shape[1:3]
    n_categories = len(np.unique(y_train))

    print(f'We have {n_training_points} training points and {n_test_points} test points')
    print(f'Each training point is an image of dimensions {input_shape}')
    print(f'Within each image, the pixels run with values from {np.amin(x_train)} to {np.amax(x_train)}')
    print(f'We see {n_categories} distinct categorical labels in the training data')

    return n_training_points, input_shape, n_categories


def view_random_image(dataset_name: str, dataset, labels, class_names, n_points: int):
    Path(f'./{dataset_name}').mkdir(parents=True, exist_ok=True)
    random_image = random.randint(1, n_points)
    class_name = class_names[labels[random_image]]

    figure = plt.figure()
    plt.imshow(dataset[random_image])
    plt.title(f'Random image {random_image} in dataset {dataset_name}')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.grid(False)
    plt.xlabel(class_name)
    plt.show()
    figure.savefig(f'./{dataset_name}/image_{random_image}.png')


def view_n_images(n_images, dataset_name: str, dataset, labels, class_names):
    Path(f'./{dataset_name}').mkdir(parents=True, exist_ok=True)
    plt_width = math.floor(math.sqrt(n_images))
    plt_height = math.ceil(n_images/plt_width)
    figure = plt.figure(figsize=(2*plt_width, 2*plt_width))

    for i in range(n_images):
        plt.subplot(plt_height, plt_width, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(dataset[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.show()
    figure.savefig(f'./{dataset_name}/first_{n_images}_images.png')


def initialise_data(dataset, dataset_name, class_names):
    x_train, y_train, x_test, y_test = load_data(dataset)
    n_training_points, input_shape, n_categories = describe_categorical_dataset(x_train, y_train, y_test)
    view_random_image(dataset_name, x_train, y_train, class_names, n_training_points)
    x_train, x_test = rescale_max_1(x_train), rescale_max_1(x_test)
    view_n_images(25, dataset_name, x_train, y_train, class_names)
    return x_train, y_train, x_test, y_test, n_training_points, input_shape, n_categories
