from midi import convert_to_midi 
import tensorflow as tf
import numpy as np
import os
import pickle


class Performer:
    def __init__(self):
        with open('encoder.pickle', 'rb') as f:
            self.encoder = pickle.load(f)
        self.model = tf.keras.models.load_model('final.h5')
        self.overfit_model = tf.keras.models.load_model('overfit.h5')

    def harmonize(self, melody):
        harmony = model.predict(melody[None])[0]
        generated_harmony = encoder.softmax_to_midi(harmony)
        original_melody = encoder.softmax_to_midi(melody[:, None, :])
        melody_and_song = [
            (melody[0], *notes)
            for melody, notes in zip(original_melody, generated_harmony)
        ]
        return melody_and_song

    def harmonize_to_file(self, melody, filepath, tempo):
        harmonies = self.harmonize(melody)
        convert_to_midi(harmonies, filepath, resolution=1 / 4, tempo=tempo)
    

def main():
    performer = Performer()
    from melodies import happy_birthday
    performer.harmonize_to_file(happy_birthday, 'happy_birthday', tempo=100)
    while True:
        pass


if __name__ == "__main__":
    main()