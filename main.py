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
)
from tensorflow.keras import Model, Sequential
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from models import BachNet, Encoder, Decoder, BahdanauAttention
from dataclasses import dataclass
import os
import json

from midi import MidiConverter
import sys
import time
from tqdm import tqdm
from collections import defaultdict
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
        self.max_song_len = max([song.shape[0] for dataset in (train, test, val) for song in dataset])
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
            [[self.one_hot_to_midi(note) for note in beat] for beat in piece]
        )

    def softmax_to_midi(self, softmax: np.ndarray, mode='argmax') -> float:
        if mode == 'beam':
            pass
        else:
            decoded = np.zeros(softmax.shape[:-1])
            for i, beat in enumerate(softmax):
                for j, voice in enumerate(beat):
                    if mode == 'argmax':
                        pred = np.argmax(voice)
                    elif mode == 'random':
                        pred = np.random.choice(self.n_notes, p=voice)
                    note = pred + self.lowest if pred != len(voice) - 1 else np.nan
                    decoded[i,j] = note
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

    def valid_range(song:np.ndarray) -> bool:
        for voice, (min_range, max_range) in vocal_ranges.items():
            if np.nanmin(song[:, voice]) < min_range or np.nanmax(song[:, voice]) > max_range:
                return False
        return True

    augmented_datasets = []
    for dataset in datasets:
        augmented_data = []
        for song in dataset:
            transpositions = filter(valid_range, map(lambda t: song + t, range(-12,12)))
            augmented_data += list(transpositions)
        augmented_datasets.append(augmented_data)

    return augmented_datasets


@dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    encoder: OneHotEncoder


def load_data(shuffle_data=False, augment=False, mode='pad') -> Dataset:
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
    if mode == 'pad':
        # Pad with rests
        for dataset in (one_hot_train, one_hot_test, one_hot_val):
            for i, piece in enumerate(dataset):
                padded = np.zeros((max_len, *piece.shape[1:]))
                padded[: piece.shape[0], :, :] = piece
                padded[piece.shape[0] :,:, -1] = 1
                dataset[i] = padded
    elif mode == 'crop':
        for dataset in (one_hot_train, one_hot_test, one_hot_val):
            for i, piece in enumerate(dataset):
                dataset[i] = piece[:min_len]
    one_hot_train = np.asarray(one_hot_train)
    one_hot_test = np.asarray(one_hot_test)
    one_hot_val = np.asarray(one_hot_val)

    x_train, x_test, x_val = [
        dataset[:,:,0,:]
        for dataset in (one_hot_train, one_hot_test, one_hot_val)
    ]
    y_train, y_test, y_val = [
        dataset[:,:,1:,:]
        for dataset in (one_hot_train, one_hot_test, one_hot_val)
    ]
    x = np.concatenate((x_train, x_test, x_val))
    y = np.concatenate((y_train, y_test, y_val))
    if shuffle_data:
        x, y = shuffle(x,y)
    return x, y, encoder


def main():
    latent_unit_count = 256
    EPOCHS = 25
    BATCH_SIZE = 8

    x, y, dataset_encoder = load_data(shuffle_data=True, augment=True, mode='crop')
    n_songs = x.shape[0]
    song_len, n_notes = x[0].shape
    validation_split = .2
    validation_idx = int(n_songs * (1 - validation_split))
    training_data = tf.data.Dataset.from_tensor_slices((x[:validation_idx], y[:validation_idx]))
    training_data = training_data.batch(BATCH_SIZE)
    validation_data = tf.data.Dataset.from_tensor_slices((x[validation_idx:], y[validation_idx:]))
    validation_data = validation_data.batch(BATCH_SIZE)

    encoder = Encoder(latent_unit_count, BATCH_SIZE)
    attention_layer = BahdanauAttention(10)
    decoder = Decoder(latent_unit_count,BATCH_SIZE,n_notes)

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction='none')

    def loss_function(real, pred):
        loss_ = loss_object(real, pred)
        return tf.reduce_mean(loss_, axis=None)

    # @tf.function # BUGGY ON MY MACHINE
    def train_step(inp, targ, enc_hidden, mode='train'):
        loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(inp, enc_hidden)
            dec_hidden = enc_hidden
            # Begin with silence?
            dec_input = tf.expand_dims([[dataset_encoder.midi_note_one_hot(np.nan)]*3] * BATCH_SIZE, 1)
            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                loss += loss_function(targ[:, t], predictions)
                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)
        batch_loss = (loss / int(targ.shape[1]))
        if mode == 'train':
            variables = encoder.trainable_variables + decoder.trainable_variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

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

    # Train
    steps_per_epoch = validation_idx // BATCH_SIZE
    val_steps_per_epoch = (n_songs - validation_idx) // BATCH_SIZE

    loss_history = defaultdict(list)
    for epoch in range(EPOCHS):
        start = time.time()
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0
        for batch, (inp, targ) in tqdm(enumerate(training_data.take(steps_per_epoch)), total=steps_per_epoch, desc=f"Training epoch {epoch+1}"):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss
            tqdm.write(f"Current loss: {total_loss/batch:.4f}")
            loss_history[epoch].append(float(batch_loss.numpy()))
        total_val_loss = 0
        with open('history.json', 'w') as out:
            json.dump(loss_history, out)
        # # Compute validation loss
        # enc_hidden = encoder.initialize_hidden_state()
        # for batch, (inp, targ) in tqdm(enumerate(validation_data.take(val_steps_per_epoch)), total=val_steps_per_epoch, desc=f"Calculating validation loss"):
        #     *_, softmax = harmonize()
        #     batch_loss = train_step(inp, targ, enc_hidden, mode='test')
        #     total_val_loss += batch_loss
        print(f"Epoch {epoch+1} Loss {total_loss/steps_per_epoch:.4f} Validation loss {total_val_loss/val_steps_per_epoch:.4f} Took {(time.time()-start):.1f}s")
        encoder.save_weights('encoder.h5')
        decoder.save_weights('decoder.h5')

    # Save weights
    enc_hidden = encoder.initialize_hidden_state()
    inp, targ = next(iter(training_data))
    train_step(inp, targ, enc_hidden)
    encoder.save_weights('encoder.h5')
    decoder.save_weights('decoder.h5')
    y_hats, *_, softmaxes = harmonize(x[0:BATCH_SIZE])
    song = dataset_encoder.softmax_to_midi(softmaxes[0], mode='argmax')
    
    midi_converter = MidiConverter()
    original_melody = dataset_encoder.softmax_to_midi(x[0][:, None, :])
    original_harmony = dataset_encoder.softmax_to_midi(y[0])
    melody_and_song = np.concatenate((original_melody, song), axis=1)
    midi_converter.convert_to_midi(melody_and_song, 'model_out', resolution=1/4, tempo=60)
    original_song = np.concatenate((original_melody, original_harmony), axis=1)
    midi_converter.convert_to_midi(original_song, 'original_out', resolution=1/4, tempo=60)
    plt.show()


if __name__ == "__main__":
    main()
