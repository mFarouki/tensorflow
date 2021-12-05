import numpy as np


def describe_categorical_dataset(x_train, y_train):
    n_training_points = y_train.shape[0]
    input_shape = x_train.shape[1:3]
    n_categories = len(np.unique(y_train))

    print(f'We have {n_training_points} training points')
    print(f'Each training point is an image of dimensions {input_shape}')
    print(f'Within each image, the pixels run with values from {np.amin(x_train)} to {np.amax(x_train)}')
    print(f'We see {n_categories} distinct categorical labels in the training data')

    return n_training_points, input_shape, n_categories