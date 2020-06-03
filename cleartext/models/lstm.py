import tensorflow as tf
import tensorflow.keras.layers as layers


def lstm(vocab_size, seq_len, units, weights=None):
    # shared embedding layer
    embed = layers.Embedding(vocab_size,
                             weights.shape[1],
                             weights=[weights],
                             input_length=seq_len,
                             trainable=False,
                             mask_zero=True)

    # encoder
    enc_in = layers.Input(shape=(seq_len,))
    enc_embed = embed(enc_in)
    enc_lstm = layers.LSTM(units, return_state=True)
    _, enc_hidden, enc_cell = enc_lstm(enc_embed)

    # decoder
    dec_in = layers.Input(shape=(seq_len,))
    dec_embed = embed(dec_in)
    dec_lstm = layers.LSTM(units, return_sequences=True, return_state=True)
    dec_out, _, _ = dec_lstm(dec_embed, initial_state=[enc_hidden, enc_cell])

    # output
    out = layers.Dense(vocab_size, activation='softmax')(dec_out)

    # model
    model = tf.keras.Model(inputs=[enc_in, dec_in], outputs=out)
    return model
