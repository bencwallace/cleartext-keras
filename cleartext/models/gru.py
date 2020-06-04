import tensorflow as tf
import tensorflow.keras.layers as layers


def build_gru(vocab_size, seq_len, units, weights=None, rec_dropout=0.2):
    # build training model
    # shared embedding layer
    embed = layers.Embedding(vocab_size,
                             weights.shape[1],
                             weights=[weights],
                             input_length=seq_len,
                             trainable=False,
                             mask_zero=True)

    # encoder
    enc_in = layers.Input(shape=(seq_len,), dtype='int32')
    enc_embed = embed(enc_in)
    enc_gru = layers.GRU(units, return_state=True, recurrent_dropout=rec_dropout)
    _, enc_state = enc_gru(enc_embed)
    # decoder
    dec_in = layers.Input(shape=(seq_len,), dtype='int32')
    dec_embed = embed(dec_in)
    dec_gru = layers.GRU(units, return_sequences=True, return_state=True)
    dec_out, _ = dec_gru(dec_embed, initial_state=enc_state)
    dec_out = layers.Dropout(0.5)(dec_out)
    # output
    out = layers.Dense(vocab_size, activation='softmax')(dec_out)
    # training model
    model = tf.keras.Model(inputs=[enc_in, dec_in], outputs=out)

    # for inference, we need the encoder
    enc_model = tf.keras.Model(inputs=enc_in, outputs=enc_state)
    # we'll also need a dummy input to fill with the initial cell state
    dec_state_in = layers.Input(shape=(units,))
    # and we need a variable lenght decoder input
    dec_in_var = layers.Input(shape=(None,))
    # we combine these elements as follows
    dec_embed_var = embed(dec_in_var)
    infer_dec_out, infer_dec_state = dec_gru(dec_embed_var, initial_state=dec_state_in)
    # this gives us the decoder
    dec_model = tf.keras.Model(inputs=[dec_in_var, dec_state_in], outputs=[infer_dec_out, infer_dec_state])

    return model, enc_model, dec_model
