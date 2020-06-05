import os

import numpy as np

from ..utils.utils import get_proj_root


def load_embedding(dim, tokenizer, embedding):
    if embedding == 'glove':
        vectors = load_glove(dim)
    vec = next(x for x in vectors.values())
    dim = vec.shape[0]

    # add 1 for padding token
    vocab_size = tokenizer.num_words + 1
    matrix = np.zeros((vocab_size, dim))
    for word, row in tokenizer.word_index.items():
        if row >= vocab_size:
            continue
        vec = vectors.get(word)
        if vec is not None:
            matrix[row] = vec

    return matrix


def load_glove(dim):
    proj_root = get_proj_root()
    glove_dir = os.path.join(proj_root, 'models/glove/')

    vectors = dict()
    with open(os.path.join(glove_dir, f'glove.6B.{dim}d.txt')) as f:
        for line in f:
            word, vec = line.split(maxsplit=1)
            vec = np.asarray(vec.split(), dtype='float32')
            vectors[word] = vec
    return vectors
