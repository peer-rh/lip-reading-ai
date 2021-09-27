from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization

from load_data import get_dataset_and_vec
from model import get_compiled_model

filepath = "checkpoints/"
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath, save_weights_only=True, save_best_only=True, verbose=1
)


def load_feature_data() -> Tuple[tf.data.Dataset, tf.data.Dataset, TextVectorization, int]:
    ds, en_vec, vocab_size = get_dataset_and_vec()
    ds = ds.shuffle(1000)
    ds_val = ds.take(len(ds)//5).batch(32)
    ds_train = ds.skip(len(ds)//5).batch(32)
    return ds_train, ds_val, en_vec, vocab_size


ds_train, ds_val, en_vec, vocab_size = load_feature_data()

model = get_compiled_model(vocab_size=vocab_size)
history = model.fit(
    ds_train,
    epochs=100,
    callbacks=[checkpoint],
    validation_data=ds_val
)
