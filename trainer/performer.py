import tensorflow as tf
import numpy as np
import os
from typing import Iterable, Tuple
import json
import sys

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
    def __init__(self, train=None, test=None, val=None, encoder_file=None):
        if encoder_file:
            with open(encoder_file, 'r') as f:
                config = json.load(f)
            self.lowest = config["lowest"]
            self.highest = config["highest"]
            self.max_song_len = config["max_song_len"]
            self.n_notes = config["n_notes"]
        else:
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

    def save_config(self, filepath):
        with open(filepath, 'w') as f:
            json.dump({
                "lowest": self.lowest, "highest": self.highest, "max_song_len": self.max_song_len, "n_notes": self.n_notes
            })

class Harmonizer:
    def __init__(self):
        self.encoder = OneHotEncoder(encoder_file="encoder_config.json")
        self.model = tf.keras.models.load_model('final.h5')
        self.overfit_model = tf.keras.models.load_model('overfit.h5')

    def harmonize(self, melody, overfit=False) -> np.ndarray:
        one_hot = self.encoder.encode_song(melody)[:,0,:]
        if overfit:
            harmony = self.overfit_model.predict(one_hot[None])[0]
        else:
            harmony = self.model.predict(one_hot[None])[0]
        generated_harmony = self.encoder.softmax_to_midi(harmony)
        melody_and_song = [
            (m[0], *notes)
            for m, notes in zip(melody, generated_harmony)
        ]
        return melody_and_song

    def harmonize_to_file(self, melody, filepath, tempo, overfit=False):
        harmonies = self.harmonize(melody, overfit)
        from app.midi import convert_to_midi
        convert_to_midi(harmonies, filepath, resolution=1 / 4, tempo=tempo)
    

def main():
    performer = Harmonizer()
    sys.exit(0)
    performer.encoder.save_config('encoder_config.json')
    from app.melodies import melodies, tempos
    for k, v in melodies.items():
        if not os.path.exists(k):
            os.mkdir(k)

        print(f"Harmonizing {k}")
        for t in range(6):
            transposed = np.asarray(v) + t
            if np.nanmax(transposed) <= performer.encoder.highest and np.nanmin(transposed) >= performer.encoder.lowest:
                performer.harmonize_to_file(transposed, f'{k}/{k}_+{t}', tempo=tempos[k])
                performer.harmonize_to_file(transposed, f'{k}/overfit_{k}_+{t}', tempo=tempos[k], overfit=True)


if __name__ == "__main__":
    main()
