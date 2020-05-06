import numpy as np
from typing import Tuple, Iterable
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from itertools import cycle

# Fix seed for reproducibility
tf.random.set_seed(0)


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
        one_hot = np.asarray([[self.midi_note_one_hot(np.random.choice(self.n_notes, p=voice)) for voice in beat]for beat in softmax])
        return self.decode_song(one_hot)


def main():
    data = np.load("data/Jsb16thSeparated.npz", allow_pickle=True, encoding="latin1")
    train, test, val = data["train"], data["test"], data["valid"]

    encoder = OneHotEncoder(train, test, val)
    one_hot_train, one_hot_test, one_hot_val = [[encoder.encode_song(x) for x in dataset] for dataset in (train, test, val)]
    
    model = tf.keras.Sequential()
    x_train, x_test, x_val = [
        [x[:, 0:1, :] for x in dataset]
        for dataset in (one_hot_train, one_hot_test, one_hot_val)
    ]
    y_train, y_test, y_val = [[y[:, 1:, :] for y in dataset]
        for dataset in (one_hot_train, one_hot_test, one_hot_val)
    ]
    input_dim = x_train[0].shape[1:]
    output_dim = y_train[0].shape[1:]
    batch_size = 8

    model.add(tf.keras.layers.LSTM(46, input_shape=(x_train[0].shape[1:]), return_sequences=True))
    model.add(tf.keras.layers.Reshape((reversed(input_dim))))
    model.add(tf.keras.layers.Dense(3))
    model.add(tf.keras.layers.Reshape(output_dim))
    model.add(tf.keras.layers.Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # We train separately on each song, but the weights are maintained.
    history = {'loss': [], 'val_loss': []}
    epochs = 10
    for epoch in range(epochs):
        for melody, harmony, val_melody, val_harmony in zip(x_train, y_train, cycle(x_val), cycle(y_val)):
            hist = model.fit(melody, harmony, epochs=1, batch_size=batch_size, validation_data=(val_melody, val_harmony))
            history['loss'] += hist.history['loss']
            history['val_loss'] += hist.history['val_loss']
            model.reset_states()
    y_hat = model.predict(x_train[0])
    song = encoder.softmax_to_midi(y_hat)
    plt.plot(history['loss'], label='Training loss')
    plt.plot(history['val_loss'], label='Validation loss')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
