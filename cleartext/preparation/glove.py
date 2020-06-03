import numpy as np


def load_glove(dim):
    with open(os.path.join('../../data/raw/glove', f'glove.6B.{dim}d.txt')) as f:
        for line in f:
            word, vec = line.split(maxsplit=1)
            vec = np.asarray(vec.split(), dtype='float32')
            embed_index[word] = vec
    return embed_index


def build_embed_matrix(embed_index, tokenizer):
    vec = next(x for x in embed_index.values())
    dim = vec.shape[0]
    
    # add 1 for padding token
    vocab_size = tokenizer.num_words + 1
    matrix = np.zeros((vocab_size, dim))
    for word, row in tokenizer.word_index.items():
        if row >= vocab_size:
            continue
        vec = embed_index.get(word)
        if vec is not None:
            matrix[row] = vec

    return matrix
