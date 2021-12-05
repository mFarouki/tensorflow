import math

import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path


def plot_image(image_number: int, predictions: np.ndarray, labels: np.ndarray, dataset: np.ndarray, class_names: list):
    true_label, image = labels[image_number], dataset[image_number]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(image, cmap=plt.cm.binary)

    prediction = predictions[image_number]
    predicted_label = np.argmax(prediction)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(prediction),
                                         class_names[true_label]),
               color=color)


def plot_value_array(image_number: int, predictions: np.ndarray, labels: np.ndarray, n_categories: int):
    prediction = predictions[image_number]
    true_label = labels[image_number]
    plt.grid(False)
    plt.xticks(range(n_categories))
    plt.yticks([])
    all_categories = plt.bar(range(n_categories), predictions[image_number], color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions[image_number])

    all_categories[predicted_label].set_color('red')
    all_categories[true_label].set_color('blue')


def view_random_image(dataset_name: str, dataset: np.ndarray, labels: np.ndarray, class_names: list,
                      predictions: np.ndarray, dataset_size: int, n_categories: int):
    Path(f'./{dataset_name}').mkdir(parents=True, exist_ok=True)
    random_image = random.randint(1, dataset_size)

    figure = plt.figure(figsize=(6, 3))
    plt.title(f'Random image {random_image} in dataset {dataset_name}')
    plt.subplot(1, 2, 1)
    plot_image(random_image, predictions, labels, dataset, class_names)
    plt.subplot(1, 2, 2)
    plot_value_array(random_image, predictions, labels, n_categories)
    plt.show()
    figure.savefig(f'./{dataset_name}/image_{random_image}_prediction.png')


def view_n_images(n_images: int, dataset_name: str, dataset: np.ndarray, labels: np.ndarray, class_names: list,
                  predictions: np.ndarray, n_categories: int):
    Path(f'./{dataset_name}').mkdir(parents=True, exist_ok=True)
    n_columns = math.floor(math.sqrt(n_images))
    n_rows = math.ceil(n_images / n_columns)
    figure = plt.figure(figsize=(4 * n_columns, 2 * n_columns))

    for i in range(n_images):
        plt.subplot(n_rows, 2 * n_columns, 2 * i + 1)
        plot_image(i, predictions, labels, dataset, class_names)
        plt.subplot(n_rows, 2 * n_columns, 2 * i + 2)
        plot_value_array(i, predictions, labels, n_categories)
    plt.tight_layout()
    plt.show()
    figure.savefig(f'./{dataset_name}/first_{n_images}_image_predictions.png')


def visualise_predictions(dataset_name: str, dataset: np.ndarray, labels: np.ndarray, class_names: list,
                          predictions: np.ndarray, n_points: int, n_categories: int, n_images: int):
    view_random_image(dataset_name, dataset, labels, class_names, predictions, n_points, n_categories)
    view_n_images(n_images, dataset_name, dataset, labels, class_names, predictions, n_categories)
