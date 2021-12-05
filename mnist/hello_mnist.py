import numpy as np
import tensorflow as tf

from clean_and_view import *
from build_model import *

dataset_name = 'mnist'
n_hidden_layer_nodes = 128
hidden_dropout_rate = 0.2
n_training_epochs = 5

class_names = ['digit 0', ' digit 1', 'digit 2', 'digit 3', 'digit 4', 'digit 5', 'digit 6', 'digit 7',
               'digit 8', ' digit 9']


def greet():
    version_check()
    h = tf.constant('Hello')
    m = tf.constant(f'{dataset_name}!')
    hm = h + " " + m
    tf.print(hm)


def main():
    greet()
    x_train, y_train, x_test, y_test, n_training_points, input_shape, n_categories = \
        initialise_data(tf.keras.datasets.mnist, dataset_name, class_names)

    model = simple_neural_net(input_shape, n_categories, n_hidden_layer_nodes, hidden_dropout_rate)

    cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # untrained model should give probability ~1/10 to each class for a given data point, as the categorical
    # cross entropy is the same as -log(p(true class))
    untrained_point_loss(n_training_points, x_train, y_train, model, cross_entropy_loss, -np.log(1 / 10))
    model = compile_model(model, cross_entropy_loss)

    model.fit(x_train, y_train, epochs=n_training_epochs)
    model.evaluate(x_test, y_test, verbose=2)


if __name__ == "__main__":
    main()
