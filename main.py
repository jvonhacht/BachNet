import numpy as np
from typing import Tuple, Iterable
from midi import MidiConverter


class OneHotEncoder:
    def __init__(self, train, test, val):
        self.lowest = int(min([min([np.min(piece) for piece in dataset]) for dataset in (train, test, val)]))
        self.highest = int(max([max([np.max(piece) for piece in dataset]) for dataset in (train, test, val)]))

    def encode_song(self, piece: Iterable[Tuple[float]]) -> np.ndarray:
        encoded = np.asarray([[self.midi_note_one_hot(note) for note in beat] for beat in piece])
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
        idx = np.nonzero(one_hot)[0].item()
        if idx == one_hot.shape[0] - 1:
            return np.nan
        return float(idx) + self.lowest

    def decode_song(self, piece: np.ndarray) -> Iterable[Tuple[float]]:
        return np.asarray([[self.one_hot_to_midi(note) for note in beat] for beat in piece])


def main():
    data = np.load("data/Jsb16thSeparated.npz", allow_pickle=True, encoding='latin1')
    train, test, val = data['train'], data['test'], data['valid']
    
    # test midi creation
    piece = train[10]
    #test3 = [(harmony[0],) for harmony in piece]
    midi_converter = MidiConverter()
    midi_converter.convert_to_midi(piece, 'test', resolution=1/16)

    encoder = OneHotEncoder(train, test, val)
    raw_song = train[0]
    raw_song[0,0] = np.nan
    one_hot_song = encoder.encode_song(raw_song)
    decoded = encoder.decode_song(one_hot_song)

    x_train = [[harmony[0] for harmony in piece] for piece in data['train']]
    y_train = [[harmony[1:] for harmony in piece] for piece in data['train']]
    x_val = [[harmony[0] for harmony in piece] for piece in data['valid']]
    y_val = [[harmony[1:] for harmony in piece] for piece in data['valid']]
    x_test = [[harmony[0] for harmony in piece] for piece in data['test']]
    y_test = [[harmony[1:] for harmony in piece] for piece in data['test']]


if __name__ == "__main__":
    main()