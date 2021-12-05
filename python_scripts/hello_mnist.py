import tensorflow as tf
import numpy as np
import random
from utilities import describe_categorical_dataset

n_hidden_layer_nodes = 128
hidden_dropout_rate = 0.2
n_training_epochs = 5


def greet():
    print(f'Running TensorFlow version {tf.__version__}')

    h = tf.constant('Hello')
    m = tf.constant('mnist!')
    hm = h + " " + m
    tf.print(hm)


def rescale_max_1(np_array):
    if np.issubdtype(np_array.dtype, np.number):
        return np_array / np.amax(np_array)
    else:
        raise ValueError("Function can only rescale numeric values")


def simple_neural_net(input_shape, n_categories, n_hidden_nodes, dropout_rate=None, activation_function='relu'):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=input_shape))
    model.add(tf.keras.layers.Dense(n_hidden_nodes, activation=activation_function))
    if dropout_rate:
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(n_categories))

    print(model.summary())
    return model


def untrained_point_loss(n_training_points, x_train, y_train, model, loss_fxn, expected_loss):
    point_to_check = random.randint(1, n_training_points)
    point_prediction = model(x_train[:point_to_check]).numpy()
    point_loss = loss_fxn(y_train[:point_to_check], point_prediction).numpy()
    print(f'The loss for data point {point_to_check} is {round(float(point_loss), 3)}, compared with an expectation of '
          f'{round(float(expected_loss), 3)} for an untrained model')


def main():
    greet()
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    n_training_points, input_shape, n_categories = describe_categorical_dataset(x_train, y_train)

    x_train, x_test = rescale_max_1(x_train), rescale_max_1(x_test)

    model = simple_neural_net(input_shape, n_categories, n_hidden_layer_nodes, hidden_dropout_rate)

    cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # untrained model should give probability ~1/10 to each class for a given data point, as the categorical
    # cross entropy is the same as -log(p(true class))
    untrained_point_loss(n_training_points, x_train, y_train, model, cross_entropy_loss, -np.log(1 / 10))

    model.compile(optimizer='adam',  # note that adam is a type of stochastic gradient descent
                            loss=cross_entropy_loss,
                            metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=n_training_epochs)
    model.evaluate(x_test,  y_test, verbose=2)


if __name__ == "__main__":
    main()
