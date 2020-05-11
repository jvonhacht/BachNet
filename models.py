import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, GRU, LSTM, Layer, Activation, RepeatVector
from tensorflow.keras import Model

class Encoder(tf.keras.Model):
  def __init__(self, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.a_gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.t_gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.b_gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    a_output, a_state = self.a_gru(x, initial_state = hidden[0])
    t_output, t_state = self.t_gru(x, initial_state = hidden[1])
    b_output, b_state = self.b_gru(x, initial_state=hidden[2])
    output = tf.stack((a_output,t_output,b_output))
    state = tf.stack((a_state,t_state,b_state))
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((3,self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.aW1 = tf.keras.layers.Dense(units)
    self.aW2 = tf.keras.layers.Dense(units)
    self.aV = tf.keras.layers.Dense(1)
    self.tW1 = tf.keras.layers.Dense(units)
    self.tW2 = tf.keras.layers.Dense(units)
    self.tV = tf.keras.layers.Dense(1)
    self.bW1 = tf.keras.layers.Dense(units)
    self.bW2 = tf.keras.layers.Dense(units)
    self.bV = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 2)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    a_score = self.aV(tf.nn.tanh(self.aW1(query_with_time_axis[0])+self.aW2(values[0])))
    t_score = self.tV(tf.nn.tanh(self.tW1(query_with_time_axis[1])+self.tW2(values[1])))
    b_score = self.bV(tf.nn.tanh(self.bW1(query_with_time_axis[2])+self.bW2(values[2])))
    
    a_weights = tf.nn.softmax(a_score, axis=1)
    t_weights = tf.nn.softmax(t_score, axis=1)
    b_weights = tf.nn.softmax(b_score, axis=1)

    a_context = tf.reduce_sum(a_weights * values[0], axis=1)
    t_context = tf.reduce_sum(t_weights * values[1], axis=1)
    b_context = tf.reduce_sum(b_weights * values[2], axis=1)

    return tf.stack((a_context,t_context,b_context)), tf.stack((a_weights,t_weights,b_weights))


class Decoder(tf.keras.Model):
  def __init__(self, dec_units, batch_sz, n_notes):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.alto = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.tenor = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.bass = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(n_notes)
    self.activation = Activation('softmax')
    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # passing the concatenated vector to the GRUs

    a, a_state = self.alto(tf.concat([tf.expand_dims(context_vector[0], 1), x[:,:,0,:]], axis=-1))
    t, t_state = self.tenor(tf.concat([tf.expand_dims(context_vector[1], 1), x[:,:,1,:]], axis=-1))
    b, b_state = self.bass(tf.concat([tf.expand_dims(context_vector[2], 1), x[:,:,2,:]], axis=-1))
    a = tf.squeeze(a, axis=1)
    t = tf.squeeze(t, axis=1)
    b = tf.squeeze(b, axis=1)

    # output shape == (batch_size * 1, hidden_size)
    # output = tf.reshape(output, (-1, output.shape[2]))
    x = tf.stack([a,t,b], axis=1)
    # output shape == (batch_size, vocab)
    x = self.fc(x)
    x = self.activation(x)
    return x, tf.stack((a_state, t_state, b_state)), attention_weights


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
