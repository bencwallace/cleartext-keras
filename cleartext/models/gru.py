import tensorflow as tf
import tensorflow.keras.layers as layers


class EncoderDecoder(tf.keras.Model):
    def __init__(self, vocab_size, embed_weights, units, seq_len):
        super().__init__()
        embed = layers.Embedding(vocab_size,
                                 embed_weights.shape[1],
                                 weights=[embed_weights],
                                 input_length=seq_len,
                                 trainable=False,
                                 mask_zero=True)
        self.encoder = Encoder(vocab_size, embed, units, seq_len)
        self.decoder = Decoder(vocab_size, embed, units, seq_len)

    def call(self, inputs, training=None):
        # inputs is source word and target word (None if not training)
        enc_in, dec_in = inputs
        _, enc_state = self.encoder(enc_in)
        dec_out = self.decoder([dec_in, enc_state], training)
        return dec_out


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embed_layer, units, seq_len):
        super().__init__()
        self.embed = embed_layer
        self.gru = layers.GRU(units, return_state=True)

    def call(self, inputs):
        # inputs is source word
        x = self.embed(inputs)
        out, state = self.gru(x)
        return out, state


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embed_layer, units, seq_len):
        super().__init__()
        self.embed = embed_layer
        self.gru = layers.GRU(units, return_sequences=True, return_state=True)
        self.fc = layers.Dense(vocab_size)

    def call(self, inputs, training=None):
        # inputs is target word (None if not training) and encoder state
        inputs, enc_state = inputs

        # todo: fix this
        # if training:
        x = self.embed(inputs)
        out, _ = self.gru(x, initial_state=enc_state)
        out = tf.reshape(out, (-1, out.shape[2]))
        out = self.fc(out)
        return out
