from typing import Dict, Tuple
import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import TextVectorization
import pickle
import numpy as np

MAX_SEQ_LEN = 160


def process_landmarks(
        landmarks: tf.Tensor,
        label: tf.Tensor,
        en_vec: TextVectorization) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    vec_label = en_vec(label)
    return {"encoder_inputs": landmarks, "decoder_inputs": vec_label[:-1]}, vec_label[1:]


def get_dataset_and_vec() -> Tuple[tf.data.Dataset, TextVectorization, int]:
    data: pd.DataFrame = pd.read_csv("data/landmarks/labels.csv",
                                     sep=":  ", names=["ids", "label"])
    file_names = "data/landmarks/" + data["ids"] + ".csv"
    labels = data["label"].str.lower()

    vocab_size = len(set(" ".join([i for i in labels]).split()))
    print(vocab_size)

    en_vec = TextVectorization(
        max_tokens=vocab_size, output_mode="int", output_sequence_length=MAX_SEQ_LEN,
    )
    en_vec.adapt(labels)

    pickle.dump({
        "config": en_vec.get_config(),
        'weights': en_vec.get_weights()}, open("en_vec.pkl", "wb"))

    landmarks = np.zeros((len(file_names), MAX_SEQ_LEN-1, 40))
    for i, filename in enumerate(file_names):
        this_landmark = np.loadtxt(filename, delimiter=", ")
        landmarks[i, -len(this_landmark):] = this_landmark

    dataset = tf.data.Dataset.from_tensor_slices((landmarks, labels))
    dataset = dataset.map(lambda x, y: process_landmarks(x, y, en_vec))
    return dataset, en_vec, vocab_size


if __name__ == "__main__":
    ds, _ = get_dataset_and_vec()
    for i in ds.take(1):
        print(i[0]["decoder_inputs"].shape)
        print(i[0]["encoder_inputs"].shape)
        print(i[1].shape)
