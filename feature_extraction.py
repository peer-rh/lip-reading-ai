import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from tqdm import tqdm


model = keras.models.load_model('ae.h5')
model.summary()
enc = keras.Model(model.get_layer("sequential").input, model.get_layer("sequential").output)

features_ds = []
labels_ds = []

def decode_img(img_path):
    img = tf.io.decode_jpeg(tf.io.read_file(img_path))
    # Resize and make grayscale
    return tf.reshape(tf.image.resize(img, [64, 64])[:,:,0], (1, 64,64,1)) / 255

data_dir = "data/prepared_data_backup/"

for i, class_name in enumerate(tqdm(os.listdir(data_dir))):
    for j, sample in enumerate(tqdm(os.listdir(os.path.join(data_dir, class_name)))):
        this_features = None
        for k, img_name in enumerate(os.listdir(os.path.join(data_dir, class_name, sample))):
            img_path = os.path.join(data_dir, class_name, sample, img_name)
            features = enc(decode_img(img_path)).numpy()
            if k == 0:
                this_features = np.array(features)
            else:
                this_features = np.concatenate((this_features, features))
        new_features = np.zeros((25, 128))
        new_features[-len(this_features):] = this_features
        features_ds.append(new_features)
        labels_ds.append(i)

features_ds = np.array(features_ds)
print(features_ds.shape)
np.save("features_ds", features_ds)
np.save("labels_ds", labels_ds)