#!/usr/bin/env python3
import os
import sys

import numpy as np

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import text

from ..models import build_lstm
from ..models.gru import build_gru
from ..preparation import load_embedding, load_data, prepare
from ..utils import get_proj_root


class Trainer(object):
    def load_data(self, vocab_size=10_000, dataset='wikismall', train_frac=0.9, num_examples='all'):
        self.vocab_size = vocab_size
        data = load_data(dataset, num_examples)
        
        # tokenize
        self.tokenizer = text.Tokenizer(num_words=vocab_size, oov_token='<UNK>', filters='')
        data, self.seq_len = prepare(data, self.tokenizer, pad=True)

        # shuffle -- todo: turn on when evaluation is set up
        num_rows = len(data)
        # data = data.sample(frac=1)

        # create array
        source_array = np.array(data['source'].tolist(), dtype='int32')
        target_array = np.array(data['target'].tolist(), dtype='int32')
        tokens_array = np.stack([source_array, target_array])

        # split
        train_size = int(train_frac * num_rows)
        train, test = tokens_array[:, :train_size, :], tokens_array[:, train_size:, :]
        self.source_in = train[0]
        self.target_in = train[1]
        self.target_out = self._left_shift(train[1])

    def load_embedding(self, dim, embedding='glove'):
        self.embed_dim = dim
        self.embed_matrix = load_embedding(dim, self.tokenizer, embedding)

    def _left_shift(self, array):
        zeros = np.zeros((array.shape[0], 1))
        return np.hstack([array[:, 1:], zeros])

    def _setup_callbacks(self):
        proj_root = get_proj_root()
        filename = '{epoch:02d}_loss_{loss:.4f}.hdf5'
        filepath = os.path.join(proj_root, 'models', self._name, filename)
        self.callbacks = [
            EarlyStopping(patience=2),
            ModelCheckpoint(filepath=filepath, save_best_only=True)
        ]


class LSTMTrainer(Trainer):
    def __init__(self):
        self._name = 'lstm'

    def build_model(self, units, dropout=0.5):
        self.units = units
        self.model = build_lstm(self.vocab_size + 1, self.seq_len, units, self.embed_matrix)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    def train(self, epochs, batch_size=32, verbose=1, validation_split=0.1):
        self._setup_callbacks()
        self.model.fit(x=[self.source_in, self.target_in],
                       y=self.target_out,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       validation_split=validation_split,
                       callbacks=self.callbacks)


class GRUTrainer(Trainer):
    def __init__(self):
        self._name = 'gru'

    def build_model(self, units):
        self.units = units
        self.model, self.enc_model, self.dec_model = build_gru(self.vocab_size + 1, self.seq_len, units, self.embed_matrix)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    def train(self, epochs, batch_size=32, verbose=1, validation_split=0.1):
        self._setup_callbacks()
        self.model.fit(x=[self.source_in, self.target_in],
                       y=self.target_out,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       validation_split=validation_split,
                       callbacks=self.callbacks)


if __name__ == '__main__':
    trainer = GRUTrainer()
    trainer.load_data(num_examples=100)
    trainer.load_embedding(50)
    trainer.build_model(100)
    trainer.train(1)
