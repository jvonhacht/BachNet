from trainer.midi import MidiConverter
from trainer.models import Encoder, Decoder
import tensorflow as tf
import numpy as np

encoder = tf.keras.models.load_model('...')
decoder = tf.keras.models.load_model('...')

def harmonize(melody):
    attention_plot = np.zeros((BATCH_SIZE, song_len, song_len))
    result = np.zeros((BATCH_SIZE, song_len, 3))
    softmax = np.zeros((BATCH_SIZE, song_len,3,n_notes))
    hidden = encoder.initialize_hidden_state()
    enc_out, enc_hidden = encoder(melody, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([[dataset_encoder.midi_note_one_hot(np.nan)]*3] * BATCH_SIZE, 1)
    for t in range(song_len):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        # storing the attention weights to plot later on
        # attention_weights = tf.reshape(attention_weights, (-1, ))
        # attention_plot[t] = attention_weights.numpy()

        predicted_idxs = tf.argmax(predictions, axis=2).numpy()
        predicted_notes = predicted_idxs + dataset_encoder.lowest
        next_input = np.zeros_like(dec_input)
        for i, song in enumerate(predicted_notes):
            for j, note in enumerate(song):
                one_hot_idx = note - dataset_encoder.lowest
                next_input[i,0,j,one_hot_idx] = 1
                if note == n_notes - 1:
                    predicted_notes[i,j] = np.nan
        result[:,t] = predicted_idxs + dataset_encoder.lowest
        softmax[:,t] = predictions
        # Feed predicted notes back into model
        dec_input = next_input
    return result, melody, attention_plot, softmax

y_hats, *_, softmaxes = harmonize(x[0:BATCH_SIZE])
song = dataset_encoder.softmax_to_midi(softmaxes[0], mode='argmax')

midi_converter = MidiConverter()
original_melody = dataset_encoder.softmax_to_midi(x[0][:, None, :])
original_harmony = dataset_encoder.softmax_to_midi(y[0])
melody_and_song = np.concatenate((original_melody, song), axis=1)
midi_converter.convert_to_midi(melody_and_song, 'model_out', resolution=1/4, tempo=60)
original_song = np.concatenate((original_melody, original_harmony), axis=1)
midi_converter.convert_to_midi(original_song, 'original_out', resolution=1/4, tempo=60)