import tensorflow as tf

import build_model
from clean_and_view import version_check, initialise_data
from evaluate_model import visualise_predictions

dataset_name = 'fashion_mnist'
n_hidden_layer_nodes = 128
n_training_epochs = 8
class_names = ['t-shirt/top', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']


def main():
    version_check()
    train_images, train_labels, test_images, test_labels, n_training_points, n_test_points, input_shape, \
        n_categories = initialise_data(tf.keras.datasets.fashion_mnist, dataset_name, class_names)
    model = build_model.simple_neural_net(input_shape, n_categories, n_hidden_layer_nodes)
    cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    build_model.compile_model(model, cross_entropy_loss)
    model.fit(train_images, train_labels, epochs=n_training_epochs)
    model.evaluate(test_images, test_labels, verbose=2)

    predictions = build_model.interpret_trained_model(model, test_images)
    visualise_predictions(dataset_name, test_images, test_labels, class_names, predictions, n_test_points,
                          n_categories, 25)


if __name__ == "__main__":
    main()
