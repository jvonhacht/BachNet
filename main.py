import numpy as np
from typing import Tuple, Iterable
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

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


def main():
    data = np.load("data/Jsb16thSeparated.npz", allow_pickle=True, encoding="latin1")
    train, test, val = data["train"], data["test"], data["valid"]

    max_len = max(map(lambda dataset: max(map(lambda x: x.shape[0], dataset)), (train, test, val)))

    dim = ((len(train), max_len) + train[0].shape[1:])
    
    padded_train = np.zeros(dim)
    for idx, piece in enumerate(train):
        padded_train[idx, :piece.shape[0], :piece.shape[1]] = piece
    x_train = padded_train[:,:,0:1]
    y_train = padded_train[:, :, 1:]
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(32, input_shape=x_train.shape[1:],return_sequences=True ))
    model.add(tf.keras.layers.Dense(3, input_shape=(x_train.shape[1:])))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(x_train, y_train, epochs=1, batch_size=8)
    y_hat = model.predict(x_train)
    pass
    # x_val = np.asarray(
    #     [np.asarray([harmony[0] for harmony in piece]) for piece in one_hot_val]
    # )
    # y_val = np.asarray(
    #     [np.asarray([harmony[1:] for harmony in piece]) for piece in one_hot_val]
    # )
    # x_test = np.asarray(
    #     [np.asarray([harmony[0] for harmony in piece]) for piece in one_hot_test]
    # )
    # y_test = np.asarray(
    #     [np.asarray([harmony[1:] for harmony in piece]) for piece in one_hot_test]
    # )


if __name__ == "__main__":
    main()
