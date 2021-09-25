import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from tensorflow.keras.layers import TextVectorization
import pickle

model = keras.models.load_model('ae.h5')
enc = keras.Model(model.get_layer("sequential").input, model.get_layer("sequential").output)

def decode_img(img_path):
    img = tf.io.decode_gif(tf.io.read_file(img_path))
    # Resize and make grayscale
    return tf.expand_dims(tf.image.resize(img, [64, 64])[:,:,:,0], 3) / 255

MAX_SEQ_LEN = 155
data = pd.read_csv("data/cropped/labels.csv", sep=":  ", names=["ids", "label"])
file_names = "data/cropped/" + data.ids + ".gif"
labels = data.label.str.lower()

vocab_size = len(set(" ".join([i for i in labels]).split()))
print(vocab_size)

dataset = tf.data.Dataset.from_tensor_slices((file_names, labels))
en_vec = TextVectorization(
    max_tokens=vocab_size, output_mode="int", output_sequence_length=MAX_SEQ_LEN,
)
en_vec.adapt(labels)

pickle.dump({
    "config": en_vec.get_config(),
    'weights': en_vec.get_weights()}
    , open("en_vec.pkl", "wb"))


def process_image(video_id, label):
    this_features = enc(decode_img(video_id))   
    paddings = [[MAX_SEQ_LEN-tf.shape(this_features)[0], 0], [0, 0]]
    features = tf.pad(this_features, paddings)
    vec_label = en_vec(label)
    return {"encoder_inputs": features, "decoder_inputs": vec_label[:-1]}, vec_label[1:]

dataset = dataset.map(process_image)

tf.data.experimental.save(dataset, "dataset.tfdata")