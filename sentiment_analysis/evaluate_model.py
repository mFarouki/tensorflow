import tensorflow as tf
import matplotlib.pyplot as plt


def plot_metric(metric_name: str, metric: list, validation_metric: list):
    epochs = range(1, len(metric) + 1)
    plt.plot(epochs, metric, 'bo', label=f'Training {metric_name}')
    plt.plot(epochs, validation_metric, 'b', label=f'Validation {metric_name}')
    plt.title(f'Training and validation {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(f'{metric_name}'.capitalize())
    plt.legend()

    plt.show()
    plt.savefig(f'./training_and_validation_{metric_name}.png')


def visualise_training(history: tf.keras.callbacks.History, model: tf.keras.Sequential, test_dataset: tf.data.Dataset):
    history_dict = history.history
    history_dict.keys()

    model.evaluate(test_dataset)

    plot_metric('loss', history_dict['loss'], history_dict['val_loss'])
    plot_metric('accuracy', history_dict['binary_accuracy'], history_dict['val_binary_accuracy'])
