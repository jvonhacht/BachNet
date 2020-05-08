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
    Bidirectional,
    Embedding,
    TimeDistributed,
    Dropout,
    RepeatVector
)
from tensorflow.keras import Model, Sequential
from tensorflow.keras import preprocessing
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

    # pad data
    max_len = min(
        map(lambda dataset: min(map(lambda x: x.shape[0], dataset)), (train, test, val))
    )
    train = [song[0:max_len] for song in train]
    test = [song[0:max_len] for song in test]
    val = [song[0:max_len] for song in val]
    """
    for dataset in (train, test, val):
        for i, piece in enumerate(dataset):
            padded = np.nan * np.ones((max_len, 4))
            padded[: piece.shape[0], :] = piece
            dataset[i] = padded
    """

    # one hot encode data
    encoder = OneHotEncoder(train, test, val)
    one_hot_train, one_hot_test, one_hot_val = [
        [encoder.encode_song(x) for x in dataset] for dataset in (train, test, val)
    ]
    x_train, x_test, x_val = [
        [x[:, 0:1, :].squeeze() for x in dataset]
        for dataset in (one_hot_train, one_hot_test, one_hot_val)
    ]
    y_train, y_test, y_val = [
        [y[:, 3:4, :].squeeze() for y in dataset]
        for dataset in (one_hot_train, one_hot_test, one_hot_val)
    ]

    # convert from shape [(seq_len, n_notes)] to (N, seq_len, n_notes)
    x_train = np.rollaxis(np.dstack(x_train), -1)
    y_train = np.rollaxis(np.dstack(y_train), -1)
    x_test = np.rollaxis(np.dstack(x_test), -1)
    y_test = np.rollaxis(np.dstack(y_test), -1)
    x_val = np.rollaxis(np.dstack(x_val), -1)
    y_val = np.rollaxis(np.dstack(y_val), -1)

    output_dim = y_train[0].shape[1:]
    n_notes = x_train[0].shape[-1]
    batch_size = 1
    latent_dim = 32

    model = Sequential()
    model.add(LSTM(latent_dim, batch_input_shape=(batch_size, max_len, n_notes), return_sequences=True, stateful=True, name='encoder'))
    model.add(TimeDistributed(Dense(n_notes)))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()

    epochs = 20
    for e in range(epochs):
        x = x_train[e % len(x_train)].reshape((1, max_len, n_notes))
        y = x_train[e % len(x_train)].reshape((1, max_len, n_notes))
        val_x = x_val[e % len(x_val)].reshape((1, max_len, n_notes))
        val_y = y_val[e % len(y_val)].reshape((1, max_len, n_notes))

        hist = model.fit(
            x,
            y,
            epochs=1,
            validation_data=(val_x, val_y),
            batch_size=batch_size,
            verbose=1,
        )
        model.reset_states()

    y_hat = model.predict(x_test[0].reshape((1, max_len, n_notes)))
    song = encoder.softmax_to_midi(y_hat).flatten()

    midi_converter = MidiConverter()
    melody_and_song = [(melody[0], harmony) for melody, harmony in zip(test[0], song)]
    midi_converter.convert_to_midi(melody_and_song, 'model_out', resolution=1/4, tempo=60)
    original_song = train[0]
    midi_converter.convert_to_midi(original_song, 'original_out', resolution=1/4, tempo=60)

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.legend()
    plt.show()

    """
    # seq2seq
    encoder_inputs = Input(shape=(None, n_notes))
    encoder = LSTM(latent_dim, return_sequences=False, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, n_notes))
    decoder = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_notes, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(loss="categorical_crossentropy", optimizer="adam")
    model.summary()
    #tf.keras.utils.plot_model(model, to_file='model.png', dpi=300, show_shapes=True, show_layer_names=True, rankdir='LR')

    print('Begin training...')
    hist = model.fit(
        [x_train, y_train],
        target_train,
        epochs=1,
        validation_data=([x_val, y_val], target_val),
        batch_size=batch_size,
        verbose=0,
    )
    print('Finish training...')
    
    # inference model
    encoder_model = Model(encoder_inputs, encoder_states)
    
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder(decoder_inputs, initial_state=decoder_state_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_state_inputs,
        [decoder_outputs] + decoder_states
    )

    print(x_train[0].shape)
    input_seq = x_train[0].reshape((1, max_len-1, 46))
    print(input_seq.shape)
    # encode to state vector
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, n_notes))
    target_seq[0, 0, 0] = 1

    stop_condition = False
    decoded_song = []
    while not stop_condition:
        output_notes, h, c = decoder_model.predict(
            [target_seq] + states_value
        )

        sampled_note_index = np.argmax(output_notes[0, -1, :])
        note = np.zeros((max_len, n_notes))
        note[sampled_note_index] = 1
        decoded_song.append(note)
        print(sampled_note_index)


    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.legend()
    plt.show()
    """

    model.save_weights("model.h5")

    """
    y_hat = model.predict(x_train[0].reshape((1, 100, 46)))
    song = encoder.softmax_to_midi(y_hat).flatten()

    midi_converter = MidiConverter()
    melody_and_song = [(melody[0], harmony) for melody, harmony in zip(train[0], song)]
    midi_converter.convert_to_midi(melody_and_song, 'model_out', resolution=1/4, tempo=60)
    original_song = train[0]
    midi_converter.convert_to_midi(original_song, 'original_out', resolution=1/4, tempo=60)
    """


if __name__ == "__main__":
    main()
