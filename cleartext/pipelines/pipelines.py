#!/usr/bin/env python3
import os
import sys

import numpy as np
import tensorflow as tf

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import text

from ..models.gru import build_gru
from ..preparation.datasets import load_data
from ..preparation.embeddings import load_embedding
from ..preparation.preprocessing import preprocess
from ..utils.utils import get_proj_root


class Pipeline(object):
    def load_data(self, num_examples=10_000, vocab_size=10_000, dataset='wikismall', train_frac=0.9):
        self.vocab_size = vocab_size
        data = load_data(dataset, num_examples)

        # tokenize
        self.tokenizer = text.Tokenizer(num_words=vocab_size, oov_token='<unk>', filters='')
        data = preprocess(data, self.tokenizer)
        self.seq_len = len(data['source'].iloc[0])

        # shuffle -- todo
        # data = data.sample(frac=1)

        # create array
        source_array = np.array(data['source'].tolist(), dtype='int32')
        target_array = np.array(data['target'].tolist(), dtype='int32')
        tokens_array = np.stack([source_array, target_array])

        # split
        num_rows = len(data)
        train_size = int(train_frac * num_rows)
        train, test = tokens_array[:, :train_size, :], tokens_array[:, train_size:, :]        
        self.train_source, self.train_target = train[0], train[1]
        self.test_source, self.test_target = test[0], test[1]

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


class GRUPipeline(Pipeline):
    def __init__(self):
        self._name = 'gru'

    def build_model(self, units):
        self.units = units
        self.model, self.enc_model, self.dec_model = build_gru(self.vocab_size + 1, self.seq_len, units, self.embed_matrix)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    def train(self, epochs, batch_size=32, verbose=1, validation_split=0.1):
        self._setup_callbacks()
        self.model.fit(x=[self.train_source, self.train_target],
                       y=self._left_shift(self.train_target),
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       validation_split=validation_split,
                       callbacks=self.callbacks)

    # todo: change greedy algorithm to beam search
    def predict_seq(self, seq, max_len=100):
        # num_padding = min(self.seq_len - len(seq) - 2, 0)
        # start_token = self.tokenizer.word_index['<start>']
        # end_token = self.tokenizer.word_index['<end>']
        # seq = [start_token, *seq, end_token, *[0] * num_padding]
        # assert len(seq) == self.seq_len, f'expected length {self.seq_len} but got {len(seq)}'
        state = self.enc_model.predict([seq])
        token = self.tokenizer.word_index['<start>']
        output = tf.constant([[token]])
        result = [token]

        for _ in tf.range(tf.constant(max_len)):
            # todo: next line is also partly responsible for warning (see next todo)
            out, state = self.dec_model.predict([output, state])

            token = np.argmax(out)
            output = tf.constant([[token]])

            # todo: next line causes insane, seemingly non-deterministic complaints about retracing
            # output = tf.concat([output, token_tensor], axis=1)

            result.append(token)

            if token == self.tokenizer.word_index['<end>']:
                break

        return result


    def evaluate(self, max_len=10):
        total_score = 0
        num_examples = self.test_source.shape[0]
        smoother = SmoothingFunction().method1
        for i in range(num_examples):
            print(f'{i} out of {num_examples}')
            candidate = self.predict_seq(self.test_source[i, :].tolist(), max_len)
            reference = self.test_target[i, :]
            total_score += sentence_bleu([reference], candidate, smoothing_function=smoother)
        return total_score / num_examples


if __name__ == '__main__':
    pipeline = GRUPipeline()
    pipeline.load_data(num_examples=100)
    pipeline.load_embedding(50)
    pipeline.build_model(100)
    pipeline.train(1)
    # print(pipeline.predict_seq([11, 21, 31] + [0] * 39))
    print(pipeline.evaluate(100))
