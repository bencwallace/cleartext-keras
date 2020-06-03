import os

import dotenv
import numpy as np


def build_embed_matrix(embed_vectors, tokenizer):
    vec = next(x for x in embed_vectors.values())
    dim = vec.shape[0]

    # add 1 for padding token
    vocab_size = tokenizer.num_words + 1
    matrix = np.zeros((vocab_size, dim))
    for word, row in tokenizer.word_index.items():
        if row >= vocab_size:
            continue
        vec = embed_vectors.get(word)
        if vec is not None:
            matrix[row] = vec

    return matrix


def load_glove(dim):
    dotenv.load_dotenv()
    proj_dir = os.getenv('PROJECT_DIR')
    glove_dir = os.path.join(proj_dir, 'data/raw/')

    embed_vectors = dict()
    with open(os.path.join(glove_dir, f'glove.6B.{dim}d.txt')) as f:
        for line in f:
            word, vec = line.split(maxsplit=1)
            vec = np.asarray(vec.split(), dtype='float32')
            embed_vectors[word] = vec
    return embed_vectors
