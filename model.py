from typing import Any
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

IMG_SIZE = 64
MAX_SEQ_LENGTH = 160
NUM_FEATURES = 40


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length: int, output_dim: int) -> None:
        super().__init__()
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # The inputs are of shape: `(batch_size, frames, num_features)`
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs: tf.Tensor, mask: tf.Tensor = None) -> tf.Tensor:
        mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
        return mask

# https://keras.io/examples/nlp/neural_machine_translation_with_transformer/


class PositionalEmbeddingText(layers.Layer):
    def __init__(self, sequence_length: int, vocab_size: int, embed_dim: int, **kwargs: Any) -> None:
        super(PositionalEmbeddingText, self).__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs: tf.Tensor, mask: tf.Tensor = None) -> tf.Tensor:
        return tf.math.not_equal(inputs, 0)


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim: int, dense_dim: int, num_heads: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation=tf.nn.gelu),
             layers.Dense(embed_dim), ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs: tf.Tensor, mask: tf.Tensor = None) -> tf.Tensor:
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def compute_mask(self, inputs: tf.Tensor, mask: tf.Tensor = None) -> tf.Tensor:
        return tf.math.not_equal(inputs, 0)


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim: int, latent_dim: int, num_heads: int, **kwargs: Any) -> None:
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(latent_dim, activation="relu"),
             layers.Dense(embed_dim), ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs: tf.Tensor, encoder_outputs: tf.Tensor, mask: tf.Tensor = None) -> tf.Tensor:
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs: tf.Tensor) -> tf.Tensor:
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1),
             tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)


def get_compiled_model(dense_dim: int = 256, num_heads: int = 8, vocab_size: int = 1354) -> keras.Model:
    sequence_len = MAX_SEQ_LENGTH
    embed_dim = NUM_FEATURES

    inputs = keras.Input(shape=(None, None), name="encoder_inputs")
    dec_inputs = keras.Input(shape=(None,), name="decoder_inputs")
    x = PositionalEmbedding(sequence_len, embed_dim)(inputs)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
    dec_pe = PositionalEmbeddingText(
        sequence_len, vocab_size, embed_dim)(dec_inputs)
    x = TransformerDecoder(embed_dim, dense_dim, num_heads)(dec_pe, x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(vocab_size, activation="softmax")(x)
    model = keras.Model([inputs, dec_inputs], outputs)

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    model = get_compiled_model()
    model.summary()
