import tensorflow as tf
from build_model import embedded_neural_net, compile_model


def train_model(train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset, max_features, patience=4,
                epochs=10):
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
    model_structure = embedded_neural_net(max_features)
    model = compile_model(model_structure)
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs, callbacks=[callback])
    return model, history
