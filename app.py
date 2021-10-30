import pickle
from typing import List

import cv2
import tensorflow as tf
from numpy.typing import NDArray
from tensorflow.keras.layers import TextVectorization

from model import get_compiled_model
from prep_data import get_face_landmarks

model = get_compiled_model()
model.load_weights("./checkpoints/checkpoint")

with open("en_vec.pkl", "rb") as f:
    data = pickle.load(f)
    en_vec = TextVectorization.from_config(data["config"])
    en_vec.set_weights(data["weights"])

output_token_string_from_index = (
    tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=en_vec.get_vocabulary(),
        mask_token='',
        invert=True))

vid = cv2.VideoCapture(0)
imgs: List[NDArray] = []
# https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/
while True:
    ret, frame = vid.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

landmarks = []
for img in imgs:
    landmarks.append(get_face_landmarks(img))

model.predict(landmarks)
print(en_vec.deco)
