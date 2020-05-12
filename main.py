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
    GRU,
    Concatenate,
)
from tensorflow.keras import Model, Sequential
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from itertools import cycle

from midi import convert_to_midi
import sys
from sklearn.utils import shuffle
import os
import json

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
        self.max_song_len = max(
            [song.shape[0] for dataset in (train, test, val) for song in dataset]
        )
        self.n_notes = self.highest - self.lowest + 1

    def encode_song(self, piece: Iterable[Tuple[float]]) -> np.ndarray:
        encoded = np.asarray(
            [[self.midi_note_one_hot(note) for note in beat] for beat in piece],
            dtype="float32",
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
            [[self.one_hot_to_midi(note) for note in beat] for beat in piece],
            dtype="float32",
        )

    def softmax_to_midi(self, softmax: np.ndarray, mode="argmax") -> float:
        if mode == "beam":
            pass
        else:
            decoded = np.zeros(softmax.shape[:-1])
            for i, beat in enumerate(softmax):
                for j, voice in enumerate(beat):
                    if mode == "argmax":
                        pred = np.argmax(voice)
                    elif mode == "random":
                        pred = np.random.choice(self.n_notes, p=voice)
                    note = pred + self.lowest if pred != len(voice) - 1 else np.nan
                    decoded[i, j] = note
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

    def valid_range(song: np.ndarray) -> bool:
        for voice, (min_range, max_range) in vocal_ranges.items():
            if (
                np.nanmin(song[:, voice]) < min_range
                or np.nanmax(song[:, voice]) > max_range
            ):
                return False
        return True

    augmented_datasets = []
    for dataset in datasets:
        augmented_data = []
        for song in dataset:
            transpositions = filter(
                valid_range, map(lambda t: song + t, range(-12, 12))
            )
            augmented_data += list(transpositions)
        augmented_datasets.append(augmented_data)

    return augmented_datasets


def load_data(shuffle_data=False, augment=False, mode="pad"):
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
    if mode == "pad":
        # Pad with rests
        for dataset in (one_hot_train, one_hot_test, one_hot_val):
            for i, piece in enumerate(dataset):
                padded = np.zeros((max_len, *piece.shape[1:]), dtype="float32")
                padded[: piece.shape[0], :, :] = piece
                padded[piece.shape[0] :, :, -1] = 1
                dataset[i] = padded
    elif mode == "crop":
        for dataset in (one_hot_train, one_hot_test, one_hot_val):
            for i, piece in enumerate(dataset):
                dataset[i] = piece[:min_len]
    one_hot_train = np.asarray(one_hot_train, dtype="float32")
    one_hot_test = np.asarray(one_hot_test, dtype="float32")
    one_hot_val = np.asarray(one_hot_val, dtype="float32")

    x_train, x_test, x_val = [
        dataset[:, :, 0, :] for dataset in (one_hot_train, one_hot_test, one_hot_val)
    ]
    y_train, y_test, y_val = [
        dataset[:, :, 1:, :] for dataset in (one_hot_train, one_hot_test, one_hot_val)
    ]
    x = np.concatenate((x_train, x_test, x_val))
    y = np.concatenate((y_train, y_test, y_val))
    if shuffle_data:
        x, y = shuffle(x, y)
    return x, y, encoder


def main():
    latent_unit_count = 1024
    EPOCHS = 500
    BATCH_SIZE = 16
    dropout_rate = .1

    output_dir = f"{latent_unit_count}_units_{BATCH_SIZE}_batch_{dropout_rate}_dropout"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    path = lambda x: os.path.join(output_dir, x)

    data_x, data_y, dataset_encoder = load_data(
        shuffle_data=True, augment=True, mode="crop"
    )
    total_n_songs = data_x.shape[0]
    validation_split = 0.2
    test_split = 0.1
    test_split_idx = int(total_n_songs * (1 - test_split))
    test_x, test_y = data_x[test_split_idx:], data_y[test_split_idx:]
    data_x, data_y = data_x[:test_split_idx], data_y[:test_split_idx]
    n_songs, song_len, n_notes = data_x.shape

    # validation_idx = int(n_songs * (1 - validation_split))
    # train_data = data_x[:validation_idx], data_y[:validation_idx]
    # val_data = data_x[validation_idx:], data_y[validation_idx:]
    
    inputs = Input(shape=(song_len, n_notes), name="melody_input")
    input_encoder = GRU(
        units=latent_unit_count,
        return_sequences=True,
        return_state=True,
        name="melody_encoder",
        dropout=dropout_rate,
    )
    x, input_state = input_encoder(inputs)
    alto_encoder = GRU(
        units=latent_unit_count,
        return_sequences=True,
        return_state=True,
        name="alto_encoder",
        dropout=dropout_rate
    )
    tenor_encoder = GRU(
        units=latent_unit_count,
        return_sequences=True,
        return_state=True,
        name="tenor_encoder",
        dropout=dropout_rate
    )
    bass_encoder = GRU(
        units=latent_unit_count,
        return_sequences=True,
        return_state=True,
        name="bass_encoder",
        dropout=dropout_rate
    )
    a, a_state = alto_encoder(x, initial_state=input_state)
    t, t_state = tenor_encoder(x, initial_state=input_state)
    b, b_state = bass_encoder(x, initial_state=input_state)
    alto_decoder = GRU(
        units=latent_unit_count, return_sequences=True, name="alto_decoder",dropout=dropout_rate
    )
    tenor_decoder = GRU(
        units=latent_unit_count, return_sequences=True, name="tenor_decoder",dropout=dropout_rate
    )
    bass_decoder = GRU(
        units=latent_unit_count, return_sequences=True, name="bass_decoder",dropout=dropout_rate
    )
    a = alto_decoder(a, initial_state=a_state)
    t = tenor_decoder(t, initial_state=t_state)
    b = bass_decoder(b, initial_state=b_state)
    reshaper = Reshape((song_len, 1, latent_unit_count))
    a = reshaper(a)
    t = reshaper(t)
    b = reshaper(b)
    x = Concatenate(axis=2)([a, t, b])
    merger = Dense(n_notes, name="fc_output")
    x = merger(x)
    outputs = Activation("softmax")(x)
    model = Model(inputs=inputs, outputs=outputs, name="BachNet")

    model.compile(loss="categorical_crossentropy", optimizer="adam")
    model.summary()
    tf.keras.utils.plot_model(
        model,
        to_file=path("model.png"),
        dpi=300,
        show_shapes=True,
        show_layer_names=True,
        rankdir="LR",
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3),
        # tf.keras.callbacks.ModelCheckpoint(
        #     filepath=path("model.{epoch:02d}-{val_loss:.2f}.h5"),
        #     monitor="val_loss",
        #     save_best_only=True,
        # ),
        # tf.keras.callbacks.TensorBoard(log_dir="./logs"),
    ]
    hist = model.fit(
        data_x,
        data_y,
        validation_split=validation_split,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        shuffle=True,
    )
    model.save(path("final_model.h5"))

    def generate_song(melody, harmony=None):
        if harmony is None:
            harmony = model.predict(melody[None])[0]
        generated_harmony = dataset_encoder.softmax_to_midi(harmony)
        original_melody = dataset_encoder.softmax_to_midi(melody[:, None, :])
        melody_and_song = [
            (melody[0], *notes)
            for melody, notes in zip(original_melody, generated_harmony)
        ]
        return melody_and_song

    training_melody, training_harmony = data_x[0], data_y[0]
    test_melody, test_harmony = test_x[0], test_y[0]

    train_actual = generate_song(training_melody, harmony=training_harmony)
    train_generated = generate_song(training_melody, harmony=None)
    test_actual = generate_song(test_melody, harmony=test_harmony)
    test_generated = generate_song(test_melody, harmony=None)

    convert_to_midi(
        train_generated, path("training_generated"), resolution=1 / 4, tempo=60
    )
    convert_to_midi(train_actual, path("training_actual"), resolution=1 / 4, tempo=60)
    convert_to_midi(test_generated, path("test_generated"), resolution=1 / 4, tempo=60)
    convert_to_midi(test_actual, path("test_actual"), resolution=1 / 4, tempo=60)

    from melodies import happy_birthday
    one_hot_happy = dataset_encoder.encode_song(happy_birthday)
    generated_happy_birthday = generate_song(one_hot_happy[:,0,:])
    convert_to_midi(generated_happy_birthday, path("happy_birthday"), resolution=1/4, tempo=120)

    plt.plot(hist.history["loss"], label="Training loss")
    plt.plot(hist.history["val_loss"], label="Validation loss")
    plt.legend()
    plt.savefig(path("history.png"), dpi=300)
    with open(path("history.json"), "w") as f:
        json.dump(
            {"loss": hist.history["loss"], "val_loss": hist.history["val_loss"]}, f
        )
    plt.show()


if __name__ == "__main__":
    main()
