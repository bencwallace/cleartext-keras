# for now omit attention to reduce number of weights
# in practice, hopefully shouldn't matter too much since sentences are pretty short

import tensorflow as tf
import tensorflow.keras.layers as layers


def lstm(vocab_size, seq_len, embed_dim, units, weights=None, train_embed=False):
    # embedding weights are trainable by default if and only if initial weights are not passed in
    train_embed = True if weights is None else train_embed

    # shared embedding layer
    embed = layers.Embedding(vocab_size,
                             embed_dim,
                             weights=[weights],
                             input_length=seq_length,
                             trainable=train_embed,
                             mask_zero=True)

    # encoder
    enc_in = layers.Input(shape=(seq_len,))
    enc_embed = embed(enc_in)
    enc_lstm = layers.LSTM(units, return_state=True)
    _, enc_hidden, enc_cell = enc_lstm(enc_embed)

    # decoder
    dec_in = Input(shape=(seq_len,))
    dec_embed = embed(dec_in)
    dec_lstm = layers.LSTM(units, return_sequences=True, return_state=True)
    dec_out, _, _ = dec_lstm(dec_embed, initial_state=[enc_hidden, enc_cell])

    # output
    out = layers.Dense(vocab_size, activation='softmax')(dec_out)

    # model
    model = tf.keras.Model(inputs=[enc_in, dec_in], outputs=out)
    return model
