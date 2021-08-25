from numpy.core.fromnumeric import shape
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow import keras
from tensorflow.python.keras.activations import get

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)


IMG_SIZE = 64
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 1024

def build_feature_extractor():
    feature_extractor = Sequential([
        layers.Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.MaxPool2D(),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(NUM_FEATURES, activation="relu")
    ])
    return feature_extractor

class FeatureExtractor(layers.Layer):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = layers.Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1))
        self.maxPool = layers.MaxPool2D()
        self.flatten = layers.Flatten()
        self.conv2 = layers.Conv2D(64, (3,3), activation='relu')
        self.conv3 = layers.Conv2D(64, (3,3), activation="relu")
        self.dense1 = layers.Dense(NUM_FEATURES, activation="relu")
    
    def call(self, x):
        n_seq = x.shape[0]
        seq_len = x.shape[1]
        x = tf.squeeze(x, 1)
        x = self.maxPool(self.conv1(x))
        x = self.maxPool(self.conv2(x))
        x = self.maxPool(self.conv3(x))
        x = self.flatten(x)
        x = self.dense1(x)
        return tf.reshape(x, (n_seq, seq_len,  NUM_FEATURES))


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim):
        super().__init__()
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
        return mask


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation=tf.nn.gelu), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        # TODO: Solve why not working
        # if mask is not None:
        #     mask = mask[:, tf.newaxis, :]
        print(inputs)
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

def get_compiled_model(dense_dim=4, num_heads=1):
    sequence_len = MAX_SEQ_LENGTH
    embed_dim = NUM_FEATURES
    classes = 10

    inputs = keras.Input(shape=(None, 64, 64, 1))
    x = FeatureExtractor()(inputs)
    x = PositionalEmbedding(sequence_len, embed_dim)(x)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model

if __name__=="__main__":
    model = get_compiled_model()
    model.summary()

    