#!/usr/bin/env python3
import sys

import numpy as np

from tensorflow.keras.preprocessing import text

from cleartext.models import lstm
from cleartext.preparation import build_embed_matrix, load_glove, load_wiki, prepare


def left_shift(array):
    zeros = np.zeros((array.shape[0], 1))
    return np.hstack([array[:, 1:], zeros])


def load_static(num_examples=1000,
                dataset='wikismall',
                vocab_size=10_000,
                embed_dim=50,
                train_frac=0.8):
    print('Preparing data')
    data = load_wiki(num_examples, dataset)
    
    # tokenize
    tokenizer = text.Tokenizer(num_words=vocab_size, oov_token='<UNK>')
    data = prepare(data, tokenizer, pad=True)

    # shuffle
    num_rows = len(data)
    data = data.sample(frac=1)

    # create array
    source_array = np.array(data['source'].tolist())
    target_array = np.array(data['target'].tolist())
    tokens_array = np.stack([source_array, target_array])

    # split
    train_size = int(train_frac * num_rows)
    train, test = tokens_array[:, :train_size, :], tokens_array[:, train_size:, :]

    print('Loading weights')
    embed_vectors = load_glove(embed_dim)
    embed_matrix = build_embed_matrix(embed_vectors, tokenizer)

    return train, test, embed_matrix


def train(epochs=10,
          batch_size=32,
          num_examples=1000,
          dataset='wikismall',
          vocab_size=10000,
          embed_dim=50,
          units=32,
          train_frac=0.8):
    train, test, embed_matrix = load_static(num_examples, dataset, vocab_size, embed_dim, train_frac)

    print('Building model')
    model = lstm(vocab_size + 1, train.shape[-1], units, embed_matrix)

    # compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    print('Training model')
    source_in = train[0]
    target_in = train[1]
    target_out = left_shift(train[1])
    model.fit(x=[source_in, target_in],
              y=target_out,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split=0.8)

    return train, test, model


if __name__ == '__main__':
    train()
