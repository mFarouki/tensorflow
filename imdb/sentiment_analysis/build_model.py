import tensorflow as tf
from tensorflow.keras import layers


cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def embedded_neural_net(max_features, embedding_dim=16, dropout_rate=0.2):
    model = tf.keras.Sequential([
        layers.Embedding(max_features + 1, embedding_dim),
        layers.Dropout(dropout_rate),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(dropout_rate),
        layers.Dense(1)])
    print(model.summary())
    return model


def compile_model(model, loss_fxn=cross_entropy_loss, optimizer='adam', metrics=None):
    # note that adam is a type of stochastic gradient descent good for multiclass problems
    if metrics is None:
        metrics = tf.metrics.BinaryAccuracy(threshold=0.0)
    model.compile(optimizer=optimizer,
                  loss=loss_fxn,
                  metrics=metrics)
    return model
