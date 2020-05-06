import numpy as np
from typing import Tuple, Iterable
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from itertools import cycle

# Fix seed for reproducibility
tf.random.set_seed(0)

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


class OneHotEncoder:
    def __init__(self, train, test, val):
        self.lowest = int(
            min(
                [
                    min([np.min(piece) for piece in dataset])
                    for dataset in (train, test, val)
                ]
            )
        )
        self.highest = int(
            max(
                [
                    max([np.max(piece) for piece in dataset])
                    for dataset in (train, test, val)
                ]
            )
        )
        self.n_notes = self.highest - self.lowest + 1

    def encode_song(self, piece: Iterable[Tuple[float]]) -> np.ndarray:
        encoded = np.asarray(
            [[self.midi_note_one_hot(note) for note in beat] for beat in piece]
        )
        return encoded

    def midi_note_one_hot(self, note: float) -> np.ndarray:
        """Convert a midi note to one-hot encoding in shape (K,)"""
        one_hot = np.zeros((self.highest - self.lowest + 1,))
        if np.isnan(note):
            return one_hot
        else:
            one_hot[int(note) - self.lowest] = 1
        return one_hot

    def one_hot_to_midi(self, one_hot: np.ndarray) -> float:
        nonzero = np.nonzero(one_hot)
        if not nonzero:
            return np.nan
        idx = nonzero[0].item()
        return float(idx) + self.lowest

    def decode_song(self, piece: np.ndarray) -> Iterable[Tuple[float]]:
        return np.asarray(
            [[self.one_hot_to_midi(note) for note in beat] for beat in piece]
        )

    def softmax_to_midi(self, softmax: np.ndarray) -> float:
        one_hot = np.asarray(
            [
                [
                    self.midi_note_one_hot(np.random.choice(self.n_notes, p=voice))
                    for voice in beat
                ]
                for beat in softmax
            ]
        )
        return self.decode_song(one_hot)


def main():
    data = np.load("data/Jsb16thSeparated.npz", allow_pickle=True, encoding="latin1")
    train, test, val = data["train"], data["test"], data["valid"]

    encoder = OneHotEncoder(train, test, val)
    one_hot_train, one_hot_test, one_hot_val = [
        [encoder.encode_song(x) for x in dataset] for dataset in (train, test, val)
    ]

    model = tf.keras.Sequential()
    x_train, x_test, x_val = [
        [x[:, 0:1, :] for x in dataset]
        for dataset in (one_hot_train, one_hot_test, one_hot_val)
    ]
    y_train, y_test, y_val = [
        [y[:, 1:, :] for y in dataset]
        for dataset in (one_hot_train, one_hot_test, one_hot_val)
    ]
    input_dim = x_train[0].shape[1:]
    output_dim = y_train[0].shape[1:]
    batch_size = 32

    model.add(
        tf.keras.layers.LSTM(
            1000, return_sequences=True, input_shape=(x_train[0].shape[1:]), activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True
        )
    )
    model.add(tf.keras.layers.LSTM(46, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True))
    model.add(tf.keras.layers.Reshape((reversed(input_dim))))
    model.add(tf.keras.layers.Dense(3))
    model.add(tf.keras.layers.Reshape(output_dim))
    model.add(tf.keras.layers.Activation("softmax"))
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    # We train separately on each song, but the weights are maintained.
    history = {"loss": [], "val_loss": []}
    epochs = 10
    for epoch in tqdm(range(epochs), desc="Epoch"):
        for i in range(len(x_train)):
            melody, harmony, val_melody, val_harmony = (
                x_train[i],
                y_train[i],
                x_val[i % len(x_val)],
                y_val[i % len(y_val)],
            )
            hist = model.fit(
                melody,
                harmony,
                epochs=5,
                batch_size=batch_size,
                validation_data=(val_melody, val_harmony),
                verbose=0,
            )
            for loss in hist.history["loss"]:
                if not history["loss"]:
                    history["loss"].append(loss)
                history["loss"].append(history["loss"][-1] * .999 + loss * .001)
            for val_loss in hist.history["val_loss"]:
                if not history["val_loss"]:
                    history["val_loss"].append(val_loss)
                history["val_loss"].append(history["val_loss"][-1]*.999+loss*.001)
            model.reset_states()
    y_hat = model.predict(x_train[0])
    song = encoder.softmax_to_midi(y_hat)
    plt.plot(history["loss"], label="Training loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
