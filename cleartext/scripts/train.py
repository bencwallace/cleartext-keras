#!/usr/bin/env python3
import sys

from tensorflow.keras.preprocessing import text

from cleartext.models import lstm
from cleartext.preparation import build_embed_matrix, load_glove, load_wiki, prepare


def train(dataset='wikismall', vocab_size=10000, embed_dim=50, units=32, train_frac=0.8):
    print('Loading data')
    data = load_wiki(dataset)

    print('Preprocessing data')
    tokenizer = text.Tokenizer(num_words=vocab_size, oov_token='<UNK>')
    data = prepare(data, tokenizer, pad=True)

    # shuffle and split into train + dev
    num_rows = len(data)
    train_size = int(train_frac * num_rows)

    data = data.sample(frac=1)
    train, dev = data[:train_size], data[train_size:]

    print('Loading weights')
    embed_vectors = load_glove(embed_dim)
    embed_matrix = build_embed_matrix(embed_vectors, tokenizer)

    print('Building model')
    model = lstm(vocab_size + 1, len(data.iloc[0, 0]), units, embed_matrix)

    # compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    print('Training model')
    source = list(train['source'])
    target = list(train['target'])
    model.fit([source, target], target, batch_size=32, verbose=2)

    return model


if __name__ == '__main__':
    train(*sys.argv[1:])
