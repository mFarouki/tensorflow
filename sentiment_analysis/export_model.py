import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses


def export_model(vectorised_layer, model):
    model_for_export = tf.keras.Sequential([vectorised_layer, model, layers.Activation('sigmoid')])

    model_for_export.compile(
        loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
    )
    return model_for_export
