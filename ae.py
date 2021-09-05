from load_data_ae import load_data
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import matplotlib.pyplot as plt


IMG_SIZE = 64
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 128

def save_img(x, epoch, step):
    for i in range(6):
        plt.subplot(330 + 1 + i)
        plt.imshow(x[i], cmap=plt.get_cmap('gray'))
    plt.savefig(f"imgs/{epoch}_{step}.jpg")

def build_encoder():
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

def build_decoder():
    decoder = Sequential([
        layers.Dense(2304, activation="relu", input_shape=(NUM_FEATURES, )),
        layers.Reshape((6,6, 64)),
        layers.UpSampling2D(),
        layers.Conv2DTranspose(64, (3,3), activation="relu"),
        layers.UpSampling2D(),
        layers.Conv2DTranspose(32, (3,3), activation="relu"),
        layers.UpSampling2D(),
        layers.Conv2DTranspose(1, (5,5), activation="sigmoid"),
    ])
    return decoder

def get_model():
    inp = keras.Input(shape=(64, 64, 1))
    x = build_encoder()(inp)
    x = build_decoder()(x)
    model = keras.Model(inp, x)
    return model

class TensorBoardImage(keras.callbacks.Callback):
    def __init__(self, model, data):
        super().__init__() 
        self.model = model
        self.data = data

    def on_batch_end(self, epoch, logs={}):
        # Load image
        img = self.data.take(1)
        # Do something to the image
        recon_img = self.model(img)
        summary = tf.Summary(value=[tf.Summary.Value(tag="Reconstructed_img", image=recon_img),
                                    tf.Summary.Value(tag="Img", image=img)])
        writer = tf.summary.FileWriter('./tensorboard_logs')
        writer.add_summary(summary, epoch)
        writer.close()

        return

def main():
    
    filepath = "checkpoints_ae/"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )
    model = get_model()

    train_ds, val_ds = load_data()
    loss_fn = keras.losses.BinaryCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss=loss_fn)
    filepath = "checkpoints_ae/"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )
    tensorboard_img = TensorBoardImage(model, val_ds)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./tensorboard_logs")


    train_ds, val_ds = load_data()

    history = model.fit(
        train_ds.batch(32),
        epochs=5,
        callbacks=[checkpoint, tensorboard_callback,tensorboard_img ],
        validation_data=val_ds.batch(32),
    )

    model.save("ae.h5")
if __name__ =="__main__":
  
    main()