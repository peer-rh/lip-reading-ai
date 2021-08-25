import tensorflow as tf
from tensorflow import keras

from load_data import load_data
from model import get_compiled_model

model = get_compiled_model()

filepath = "/tmp/video_classifier"
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath, save_weights_only=True, save_best_only=True, verbose=1
)

train_ds, val_ds = load_data()

model = get_compiled_model()
history = model.fit(
    train_ds,
    epochs=5,
    callbacks=[checkpoint],
    validation_data=val_ds,
)