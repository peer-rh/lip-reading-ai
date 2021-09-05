import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.python.ops.gen_array_ops import unique

from model import get_compiled_model

# model = get_compiled_model()

filepath = "checkpoints/"
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath, save_weights_only=True, save_best_only=True, verbose=1
)

def load_feature_data():
    features = np.load("features_ds.npy")
    print(features.shape)
    labels = np.load("labels_ds.npy")
    labels = labels.reshape(-1, 1)
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    ds = ds.shuffle(1000)
    ds_val = ds.take(len(ds)//5).batch(32)
    ds_train = ds.skip(len(ds)//5).batch(32)
    return ds_train, ds_val

ds_train, ds_val = load_feature_data()

model = get_compiled_model()
history = model.fit(
    ds_train,
    epochs=100,
    callbacks=[checkpoint],
    validation_data=ds_val
) 