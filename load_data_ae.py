import tensorflow as tf
import os
import pathlib
import numpy as np

data_path = "data/prepared_data_backup/"
data_dir = pathlib.Path(data_path)
AUTOTUNE = tf.data.AUTOTUNE

class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))

# def get_label(file_path):
#     parts = tf.strings.split(file_path, os.path.sep)
#     one_hot = parts[-2] == class_names
#     return tf.argmax(one_hot)

def decode_img(img):
    img = tf.io.decode_jpeg(img)
    # Resize and make grayscale
    return tf.expand_dims(tf.image.resize(img, [64, 64])[:,:,0], axis=2) / 255

def process_path(file_path):
    # label = get_label(file_path)
    img = decode_img(tf.io.read_file(file_path))
    return img, img

def load_data():
    list_ds = tf.data.Dataset.list_files(data_path + "*/*/*", shuffle=False)
    list_ds = list_ds.shuffle(len(list_ds), reshuffle_each_iteration=False)
    ds_size = len(list_ds)
    val_size = int(ds_size * 0.2)
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)
    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    return train_ds, val_ds

if __name__=="__main__":
    train_ds, val_ds = load_data()
    for image, label in train_ds.take(1):
        print("image", image.numpy().shape)
        print("Label: ", label.numpy())