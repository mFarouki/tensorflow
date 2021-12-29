import sys
import random
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers

sys.path.append('../')
from utilities.utility_functions import display_info


def explore(vocabulary_size=1000, dimensions=5):
    embedding_layer = layers.Embedding(vocabulary_size, dimensions)
    view_random_embeddings(embedding_layer, vocabulary_size)


def view_random_embeddings(embedding_layer: layers.Embedding, vocabulary_size: int, n_embeddings=3):
    values_to_view = random.choices(range(vocabulary_size), k=n_embeddings)
    result = embedding_layer(tf.constant(values_to_view))
    n_dimensions = result.numpy().shape[1]
    display_info(f'Viewing {n_embeddings} random embeddings')
    print(result.numpy())
    ax = sns.heatmap(result.numpy(), center=0)
    ax.tick_params(bottom=False, left=False)
    ax.set_xticklabels([f'Dimension {str(value)}' for value in range(n_dimensions)], rotation=60)
    ax.set_yticklabels([f'Word {str(value)}' for value in values_to_view], rotation=0)
    plt.title(f'Embeddings for {n_embeddings} random words')
    plt.tight_layout()
    plt.show()
    plt.savefig('sample_embeddings.png')
