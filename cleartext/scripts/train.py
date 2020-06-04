#!/usr/bin/env python3
import os
import sys

import numpy as np

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import text

from ..models import lstm
from ..preparation import load_embedding, load_data, prepare
from ..utils import get_proj_root


class LSTMTrainer(object):
    def load_data(self, vocab_size, dataset='wikilarge', train_frac=0.9, num_examples='all'):
        self.vocab_size = vocab_size
        data = load_data(dataset, num_examples)
        
        # tokenize
        self.tokenizer = text.Tokenizer(num_words=vocab_size, oov_token='<UNK>')
        data, self.seq_len = prepare(data, self.tokenizer, pad=True)

        # shuffle -- todo: turn on when evaluation is set up
        num_rows = len(data)
        # data = data.sample(frac=1)

        # create array
        source_array = np.array(data['source'].tolist())
        target_array = np.array(data['target'].tolist())
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

    def build_model(self, units, dropout=0.5):
        self.model = lstm(self.vocab_size + 1, self.seq_len, units, self.embed_matrix)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        self._setup_callbacks(units)

    def train(self, epochs, batch_size=32, verbose=1, validation_split=0.8):
        self.model.fit(x=[self.source_in, self.target_in],
                       y=self.target_out,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       validation_split=validation_split,
                       callbacks=self.callbacks)

    def _setup_callbacks(self, units):
        proj_root = get_proj_root()
        filepath = os.path.join(proj_root, 'models/lstm/lstm_epoch_{epoch:02d}_loss_{loss:.4f}.hdf5')
        self.callbacks = [
            EarlyStopping(patience=2),
            ModelCheckpoint(filepath=filepath, save_best_only=True)
        ]

    def _left_shift(self, array):
        zeros = np.zeros((array.shape[0], 1))
        return np.hstack([array[:, 1:], zeros])
