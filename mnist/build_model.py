import tensorflow as tf
import random


def simple_neural_net(input_shape, n_categories, n_hidden_nodes, dropout_rate=None, activation_function='relu'):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=input_shape))
    model.add(tf.keras.layers.Dense(n_hidden_nodes, activation=activation_function))
    if dropout_rate:
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(n_categories))

    print(model.summary())
    return model


def compile_model(model, loss_fxn, optimizer='adam', metrics=None):
    # note that adam is a type of stochastic gradient descent good for multiclass problems
    if metrics is None:
        metrics = ['accuracy']
    model.compile(optimizer=optimizer,
                  loss=loss_fxn,
                  metrics=metrics)
    return model


def untrained_point_loss(n_training_points, x_train, y_train, model, loss_fxn, expected_loss):
    point_to_check = random.randint(1, n_training_points)
    point_prediction = model(x_train[:point_to_check]).numpy()
    point_loss = loss_fxn(y_train[:point_to_check], point_prediction).numpy()
    print(f'The loss for data point {point_to_check} is {round(float(point_loss), 3)}, compared with an expectation of '
          f'{round(float(expected_loss), 3)} for an untrained model')


def interpret_trained_model(model, dataset):
    # add softmax layer to convert logits to probabilities - should only be added after model is trained
    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])
    return probability_model.predict(dataset)
