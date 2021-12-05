import tensorflow as tf
import numpy as np
import random

print(f'Running TensorFlow version {tf.__version__}')

h = tf.constant('Hello')
m = tf.constant('mnist!')
hm = h + " " + m
tf.print(hm)


def get_dataset_information(x_train, y_train):
    n_training_points = y_train.shape[0]
    input_shape = x_train.shape[1:3]
    n_categories = len(np.unique(y_train))

    print(f'We have {n_training_points} training points')
    print(f'Each training point is an image of dimensions {input_shape}')
    print(f'Within each image, the pixels run with values from {np.amin(x_train)} to {np.amax(x_train)}')
    print(f'We see {n_categories} distinct categorical labels in the training data')

    return n_training_points, input_shape, n_categories


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
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    n_training_points, input_shape, n_categories = get_dataset_information(x_train, y_train)

    x_train, x_test = rescale_max_1(x_train), rescale_max_1(x_test)

    # train a simple neural net with 128 hidden layers and a dropout layer with a dropout rate of 0.2
    model = simple_neural_net(input_shape, n_categories, 128, 0.2)

    cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # untrained model should give probability ~1/10 to each class for a given data point. as the categorical
    # cross entropy is the same as -log(p(true class))
    untrained_point_loss(n_training_points, x_train, y_train, model, cross_entropy_loss, -np.log(1 / 10))

    model.compile(optimizer='adam',  # note that adam is a type of stochastic gradient descent
                            loss=cross_entropy_loss,
                            metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test,  y_test, verbose=2)


if __name__ == "__main__":
    main()
