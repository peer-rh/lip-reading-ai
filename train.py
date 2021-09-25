import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.python.ops.gen_array_ops import unique
from tensorflow.keras.layers import TextVectorization
import pickle

from model import get_compiled_model

# model = get_compiled_model()

filepath = "checkpoints/"
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath, save_weights_only=True, save_best_only=True, verbose=1
)

def load_feature_data():
    ds = tf.data.experimental.load("./dataset.tfdata")
    ds = ds.shuffle(1000)
    ds_val = ds.take(len(ds)//5).batch(32)
    ds_train = ds.skip(len(ds)//5).batch(32)
    return ds_train, ds_val


def load_vec():
    from_disk = pickle.load(open("en_vec.pkl", "rb"))
    new_v = TextVectorization.from_config(from_disk['config'])
    new_v.set_weights(from_disk['weights'])
    return new_v

ds_train, ds_val = load_feature_data()
en_vec = load_vec()

model = get_compiled_model()
history = model.fit(
    ds_train,
    epochs=100,
    callbacks=[checkpoint],
    validation_data=ds_val
) 