import numpy as np
from typing import Tuple, Iterable
import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM,
    Reshape,
    Dense,
    Activation,
    RepeatVector,
    Input,
)
from tensorflow.keras import Model, Sequential
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from itertools import cycle

from midi import MidiConverter
import sys
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
                    min([np.nanmin(piece) for piece in dataset])
                    for dataset in (train, test, val)
                ]
            )
        )
        self.highest = int(
            max(
                [
                    max([np.nanmax(piece) for piece in dataset])
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

    max_len = max(
        map(lambda dataset: max(map(lambda x: x.shape[0], dataset)), (train, test, val))
    )
    # for dataset in (train, test, val):
    #     for i, piece in enumerate(dataset):
    #         padded = np.nan * np.ones((max_len, 4))
    #         padded[: piece.shape[0], :] = piece
    #         dataset[i] = padded

    encoder = OneHotEncoder(train, test, val)
    one_hot_train, one_hot_test, one_hot_val = [
        [encoder.encode_song(x) for x in dataset] for dataset in (train, test, val)
    ]

    x_train, x_test, x_val = [
        [x[:, 0:1, :] for x in dataset]
        for dataset in (one_hot_train, one_hot_test, one_hot_val)
    ]
    y_train, y_test, y_val = [
        [y[:, 1:, :] for y in dataset]
        for dataset in (one_hot_train, one_hot_test, one_hot_val)
    ]
    output_dim = y_train[0].shape[1:]
    n_notes = x_train[0].shape[-1]
    batch_size = max_len

    latent_unit_count = 1024
    model = Sequential()
    model.add(LSTM(units=latent_unit_count, input_shape=(1, n_notes), name='encoder'))
    # value mode
    model.add(Reshape((1, latent_unit_count)))
    model.add(LSTM(units=latent_unit_count, return_sequences=True, name='decoder'))
    # sequence mode
    model.add(Dense(n_notes * 3))
    model.add(Reshape((3, n_notes)))
    model.add(Activation('softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    model.summary()
    tf.keras.utils.plot_model(model, to_file='model.png', dpi=300, show_shapes=True, show_layer_names=True, rankdir='LR')

    # We train separately on each song, but the weights are maintained.
    history = {"loss": [], "val_loss": []}
    epochs = 5
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
                validation_data=(val_melody, val_harmony),
                batch_size=max_len,
                verbose=0,
            )
            for loss in hist.history["loss"]:
                if not history["loss"]:
                    history["loss"].append(loss)
                history["loss"].append(history["loss"][-1] * .999 + loss * .001)
            for val_loss in hist.history["val_loss"]:
                if not history["val_loss"]:
                    history["val_loss"].append(val_loss)
                history["val_loss"].append(history["val_loss"][-1]*.999+val_loss*.001)
            model.reset_states()
    model.save_weights("model.h5")

    y_hat = model.predict(x_train[0])
    song = encoder.softmax_to_midi(y_hat)

    midi_converter = MidiConverter()
    melody_and_song = [(melody[0], notes[0], notes[1], notes[2]) for melody, notes in zip(train[0], song)]
    midi_converter.convert_to_midi(melody_and_song, 'model_out', resolution=1/4, tempo=60)
    original_song = train[0]
    midi_converter.convert_to_midi(original_song, 'original_out', resolution=1/4, tempo=60)

    plt.plot(history["loss"], label="Training loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
