import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, GRU, LSTM, Layer, Activation, RepeatVector
from tensorflow.keras import Model

class BachNet(Model):
    def __init__(self, latent_units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_units = latent_units
        

    def build(self, input_shape):
        self.encoder = LSTM(units=self.latent_units, input_shape=input_shape)
        self.encoder_repeater = RepeatVector(input_shape[1])
        self.alto = LSTM(units=self.latent_units, return_sequences=True)
        self.tenor = LSTM(units=self.latent_units, return_sequences=True)
        self.bass = LSTM(units=self.latent_units, return_sequences=True)
        self.dense = Dense(input_shape[2])
        self.activation = Activation('softmax')

    def call(self, input_tensor, training=False):
        x = self.encoder(input_tensor)
        x = self.encoder_repeater(x)
        a = self.alto(x)
        t = self.tenor(x)
        b = self.bass(x)
        x = tf.stack([a, t, b], axis=2)
        x = self.dense(x)
        return self.activation(x)
