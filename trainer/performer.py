from midi import convert_to_midi 
from task import load_data
import tensorflow as tf
import numpy as np
import os

def main():
    data_x, data_y, dataset_encoder = load_data(
        gcp=False, shuffle_data=True, augment=True, mode="crop"
    )

    total_n_songs = data_x.shape[0]
    test_split = 0.1
    test_split_idx = int(total_n_songs * (1 - test_split))
    test_x, test_y = data_x[test_split_idx:], data_y[test_split_idx:]
    data_x, data_y = data_x[:test_split_idx], data_y[:test_split_idx]
    n_songs, song_len, n_notes = data_x.shape

    model_path = '1024_units_16_batch_0.1_dropout_final_model.h5'
    output_dir = 'latest'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = tf.keras.models.load_model(model_path)
    path = lambda x: os.path.join(output_dir, x)

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

if __name__ == "__main__":
    main()