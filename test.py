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
    Dropout,
)
from tensorflow.keras import Model, Sequential
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from itertools import cycle

from enum import Enum, auto
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


class _EncoderNoteType(Enum):
    NOTE = auto()
    REST = auto()
    REPEAT = auto()


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
        self.n_notes = self.highest - self.lowest + 1 + 2

    def encode_song(self, piece: Iterable[Tuple[float]]) -> np.ndarray:
        encoded = []
        previous_notes = [np.nan] * 4
        for beat in piece:
            encoded_beat = []
            for i, (note, previous) in enumerate(zip(beat, previous_notes)):
                if note == previous:
                    encoded_beat.append(self.midi_note_one_hot(note, _EncoderNoteType.REPEAT))
                else:
                    previous_notes[i] = note
                    encoded_beat.append(self.midi_note_one_hot(note, note_type=_EncoderNoteType.REST if np.isnan(note) else _EncoderNoteType.NOTE))
            encoded.append(encoded_beat)
        return np.asarray(encoded)

    def midi_note_one_hot(self, note: float, note_type: _EncoderNoteType) -> np.ndarray:
        """Convert a midi note to one-hot encoding in shape (K,)"""
        one_hot = np.zeros((self.highest - self.lowest + 1 + 2,))
        if note_type is _EncoderNoteType.REST:
            one_hot[0] = 1
        elif note_type is _EncoderNoteType.NOTE:
            one_hot[int(note) - self.lowest] = 1
        elif note_type is _EncoderNoteType.REPEAT:
            one_hot[-1] = 1
        else:
            raise SyntaxError("Unsupported note type")
        return one_hot

    def one_hot_to_midi(self, one_hot: np.ndarray):
        nonzero = np.nonzero(one_hot)
        if not nonzero:
            return np.nan, _EncoderNoteType.REST
        idx = nonzero[0].item()
        if idx == len(one_hot) - 1:
            # Final index is repeat
            return idx, _EncoderNoteType.REPEAT
        if idx == 0:
            return np.nan, _EncoderNoteType.REST
        return float(idx) + self.lowest + 1, _EncoderNoteType.NOTE

    def decode_song(self, piece: np.ndarray) -> Iterable[Tuple[float]]:
        decoded = []
        previous_notes = [np.nan] * 4
        for beat in piece:
            decoded_beat = []
            for i, (one_hot, previous) in enumerate(zip(beat, previous_notes)):
                note, note_type = self.one_hot_to_midi(one_hot)
                if note_type == _EncoderNoteType.REPEAT:
                    decoded_beat.append(previous)
                elif note_type == _EncoderNoteType.REST:
                    decoded_beat.append(np.nan)
                elif note_type == _EncoderNoteType.NOTE:
                    previous_notes[i] = note
                    decoded_beat.append(note)
            decoded.append(decoded_beat)
        return np.asarray(decoded)

    def softmax_to_midi(self, softmax: np.ndarray) -> float:
        song = []
        previous_notes = [np.nan] * 4
        for beat in softmax:
            notes = [np.random.choice(self.n_notes, p=voice) for voice in beat]
            for i, (note, previous) in enumerate(zip(notes, previous_notes)):
                if note == 0:
                    notes[i] = np.nan
                elif note == self.n_notes - 1:
                    notes[i] = previous
                else:
                    note = note + self.lowest + 1
                    previous_notes[i] = note
                    notes[i] = note
            song.append(notes)
        return np.asarray(song)



def main():
    data = np.load("data/Jsb16thSeparated.npz", allow_pickle=True, encoding="latin1")
    train, test, val = data["train"], data["test"], data["valid"]

    max_len = max(
        map(lambda dataset: max(map(lambda x: x.shape[0], dataset)), (train, test, val))
    )    
    for dataset in (train, test, val):
        for i, piece in enumerate(dataset):
            padded = np.nan * np.ones((max_len, 4))
            padded[: piece.shape[0], :] = piece
            dataset[i] = padded

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
    
    print(x_train[0].shape)

    dropout = True
    epochs = 5
    #latent_unit_count = 256
    latent_unit_count = 512

    output_dim = y_train[0].shape[1:]
    n_notes = x_train[0].shape[-1]    
    batch_size = max_len    

    model = Sequential()
    model.add(LSTM(units=latent_unit_count, input_shape=(1, n_notes), name='encoder'))
    if(dropout):
        model.add(Dropout(0.3))    
    model.add(Dense(n_notes * 3))
    model.add(Reshape((3, n_notes)))
    model.add(Activation('softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    model.summary()
    
    for epoch in tqdm(range(epochs), desc='epoch'):
        for i in tqdm(range(len(x_train)), desc='i'):
            melody, harmony, val_melody, val_harmony = (
                x_train[i],
                y_train[i],
                x_val[i % len(x_val)],
                y_val[i % len(y_val)],
            )
            batch_size = melody.shape[0]
            hist = model.fit(
                melody,
                harmony,
                epochs=1,
                validation_data=(val_melody, val_harmony),
                batch_size=32,
                verbose=0,
            )
            model.reset_states()
        
    model.save_weights(f"{latent_unit_count}_units_model.h5")
    y_hat = model.predict(x_train[0])
    song = encoder.softmax_to_midi(y_hat)

    midi_converter = MidiConverter()
    melody_and_song = [(melody[0], notes[0], notes[1], notes[2]) for melody, notes in zip(train[0], song)]
    midi_converter.convert_to_midi(melody_and_song, f'{latent_unit_count}_units_model_out', resolution=1/4, tempo=60)
    original_song = train[0]
    midi_converter.convert_to_midi(original_song, f'{latent_unit_count}_original_out', resolution=1/4, tempo=60)


if __name__ == '__main__':
    main()