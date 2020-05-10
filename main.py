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
from sklearn.utils import shuffle
from models import BachNet
from dataclasses import dataclass
import os

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
        self.max_song_len = max([song.shape[0] for dataset in (train, test, val) for song in dataset])
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
            one_hot[-1] = 1
        else:
            one_hot[int(note) - self.lowest] = 1
        return one_hot

    def one_hot_to_midi(self, one_hot: np.ndarray) -> float:
        if one_hot[-1] == 1:
            return np.nan
        nonzero = np.nonzero(one_hot)
        idx = nonzero[0].item()
        return float(idx) + self.lowest

    def decode_song(self, piece: np.ndarray) -> Iterable[Tuple[float]]:
        return np.asarray(
            [[self.one_hot_to_midi(note) for note in beat] for beat in piece]
        )

    def softmax_to_midi(self, softmax: np.ndarray, mode='argmax') -> float:
        if mode == 'beam':
            pass
        else:
            decoded = np.zeros(softmax.shape[:-1])
            for i, beat in enumerate(softmax):
                for j, voice in enumerate(beat):
                    if mode == 'argmax':
                        pred = np.argmax(voice)
                    elif mode == 'random':
                        pred = np.random.choice(self.n_notes, p=voice)
                    note = pred + self.lowest if pred != len(voice) - 1 else np.nan
                    decoded[i,j] = note
        return decoded


def augment_data(*datasets):
    all_songs = np.concatenate(datasets)
    all_tones = {
        "s": [x for song in all_songs for x in song[:, 0]],
        "a": [x for song in all_songs for x in song[:, 1]],
        "t": [x for song in all_songs for x in song[:, 2]],
        "b": [x for song in all_songs for x in song[:, 3]],
    }
    soprano_range = np.nanmin(all_tones["s"]), np.nanmax(all_tones["s"])
    alto_range = np.nanmin(all_tones["a"]), np.nanmax(all_tones["a"])
    tenor_range = np.nanmin(all_tones["t"]), np.nanmax(all_tones["t"])
    bass_range = np.nanmin(all_tones["b"]), np.nanmax(all_tones["b"])
    vocal_ranges = {0: soprano_range, 1: alto_range, 2: tenor_range, 3: bass_range}

    def valid_range(song:np.ndarray) -> bool:
        for voice, (min_range, max_range) in vocal_ranges.items():
            if np.nanmin(song[:, voice]) < min_range or np.nanmax(song[:, voice]) > max_range:
                return False
        return True

    augmented_datasets = []
    for dataset in datasets:
        augmented_data = []
        for song in dataset:
            transpositions = filter(valid_range, map(lambda t: song + t, range(-12,12)))
            augmented_data += list(transpositions)
        augmented_datasets.append(augmented_data)

    return augmented_datasets


@dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    encoder: OneHotEncoder


def load_data(shuffle_data=False, augment=False, mode='pad') -> Dataset:
    data = np.load("data/Jsb16thSeparated.npz", allow_pickle=True, encoding="latin1")
    train, test, val = data["train"], data["test"], data["valid"]

    max_len = max(
        map(lambda dataset: max(map(lambda x: x.shape[0], dataset)), (train, test, val))
    )
    min_len = min(
        map(lambda dataset: min(map(lambda x: x.shape[0], dataset)), (train, test, val))
    )
    # for dataset in (train, test, val):
    #     for i, piece in enumerate(dataset):
    #         padded = np.nan * np.ones((max_len, 4))
    #         padded[: piece.shape[0], :] = piece
    #         dataset[i] = padded
    if augment:
        train, test, val = augment_data(train, test, val)
    encoder = OneHotEncoder(train, test, val)

    one_hot_train, one_hot_test, one_hot_val = [
        [encoder.encode_song(x) for x in dataset] for dataset in (train, test, val)
    ]
    # Cut to min length
    if mode == 'pad':
        # Pad with rests
        for dataset in (one_hot_train, one_hot_test, one_hot_val):
            for i, piece in enumerate(dataset):
                padded = np.zeros((max_len, *piece.shape[1:]))
                padded[: piece.shape[0], :, :] = piece
                padded[piece.shape[0] :,:, -1] = 1
                dataset[i] = padded
    elif mode == 'crop':
        for dataset in (one_hot_train, one_hot_test, one_hot_val):
            for i, piece in enumerate(dataset):
                dataset[i] = piece[:min_len]
    one_hot_train = np.asarray(one_hot_train)
    one_hot_test = np.asarray(one_hot_test)
    one_hot_val = np.asarray(one_hot_val)

    x_train, x_test, x_val = [
        dataset[:,:,0,:]
        for dataset in (one_hot_train, one_hot_test, one_hot_val)
    ]
    y_train, y_test, y_val = [
        dataset[:,:,1:,:]
        for dataset in (one_hot_train, one_hot_test, one_hot_val)
    ]
    x = np.concatenate((x_train, x_test, x_val))
    y = np.concatenate((y_train, y_test, y_val))
    if shuffle_data:
        x, y = shuffle(x,y)
    return x, y, encoder


def main():
    x, y, encoder = load_data(shuffle_data=True, augment=True, mode='crop')
    output_dim = y[0].shape[1:]
    n_notes = x[0].shape[-1]

    latent_unit_count = 1024
    model = BachNet(latent_unit_count)
    _ = model(x[0:1])
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    model.summary()
    tf.keras.utils.plot_model(model, to_file='model.png', dpi=300, show_shapes=True, show_layer_names=True, rankdir='LR', expand_nested=True)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    callbacks = [
        # tf.keras.callbacks.EarlyStopping(patience=5),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir+'/model.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss'),
        # tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    ]
    EPOCHS = 100
    BATCH_SIZE = 64
    validation_split = .2
    validation_idx = int(x.shape[0] * (1 - validation_split))
    training_data = tf.data.Dataset.from_tensor_slices((x[:validation_idx], y[:validation_idx]))
    training_data = training_data.batch(BATCH_SIZE)
    validation_data = tf.data.Dataset.from_tensor_slices((x[validation_idx:], y[validation_idx:]))
    validation_data = validation_data.batch(BATCH_SIZE)

    # model.load_weights('model.11-1.40.h5')
    hist = model.fit(training_data, callbacks=callbacks, epochs=EPOCHS, validation_data=validation_data)
    model.save_weights("model.h5")

    plt.plot(hist.history["loss"], label="Training loss")
    plt.plot(hist.history["val_loss"], label="Validation loss")
    plt.legend()
    plt.savefig('model_loss.png', dpi=300)

    y_hat = model.predict(x[0:1])[0]
    song = encoder.softmax_to_midi(y_hat, mode='argmax')
    
    midi_converter = MidiConverter()
    original_melody = encoder.softmax_to_midi(x[0][:, None, :])
    original_harmony = encoder.softmax_to_midi(y[0])
    melody_and_song = np.concatenate((original_melody, song), axis=1)
    midi_converter.convert_to_midi(melody_and_song, 'model_out', resolution=1/4, tempo=60)
    original_song = np.concatenate((original_melody, original_harmony), axis=1)
    midi_converter.convert_to_midi(original_song, 'original_out', resolution=1/4, tempo=60)
    plt.show()


if __name__ == "__main__":
    main()
