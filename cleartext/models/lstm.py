import tensorflow as tf
import tensorflow.keras.layers as layers


def build_lstm(vocab_size, seq_len, units, weights=None, rec_dropout=0.2):
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
    enc_lstm = layers.LSTM(units, return_state=True, recurrent_dropout=rec_dropout)
    _, enc_hidden, enc_cell = enc_lstm(enc_embed)

    # decoder
    dec_in = layers.Input(shape=(seq_len,), dtype='int32')
    dec_embed = embed(dec_in)
    dec_lstm = layers.LSTM(units, return_sequences=True, return_state=True)
    dec_out, _, _ = dec_lstm(dec_embed, initial_state=[enc_hidden, enc_cell])
    dec_out = layers.Dropout(0.5)(dec_out)

    # output
    out = layers.Dense(vocab_size, activation='softmax')(dec_out)

    # model
    training_model = tf.keras.Model(inputs=[enc_in, dec_in], outputs=out)
    
    # build inference model
    # todo: accept one-hot vectors instead
    # infer_enc_in = layers.Input(shape=(None,), dtype='int32')
    # infer_enc_embed = embed(infer_enc_in)    
    # inference_model = None

    # return training_model, inference_model
    return training_model
