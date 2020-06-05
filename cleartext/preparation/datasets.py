import io
import os

import pandas as pd

from ..utils.utils import get_proj_root


def load_data(dataset, num_examples='all'):
    if dataset in ['wikismall', 'wikilarge']:
        return load_wiki(dataset, num_examples)


def load_wiki(dataset, num_examples):
    proj_root = get_proj_root()
    wiki_dir = os.path.join(proj_root, 'data/raw/data-simplification', dataset)

    prefix = 'PWKP_108016.tag.80.aner.ori' if dataset == 'wikismall' else 'wiki.full.aner.ori'
    data = []
    for split in ['train', 'valid', 'test']:
        for loc in ['src', 'dst']:
            file_name = '.'.join([prefix, split, loc])
            file_path = os.path.join(wiki_dir, file_name)
            stream = io.open(file_path)
            lines = stream.read().split('\n')
            data.append(lines)

    src_train, dst_train, src_valid, dst_valid, src_test, dst_test = data
    train = pd.DataFrame(zip(src_train, dst_train), columns=['source', 'target'])
    valid = pd.DataFrame(zip(src_valid, dst_valid), columns=['source', 'target'])
    test = pd.DataFrame(zip(src_test, dst_test), columns=['source', 'target'])

    data = pd.concat([train, valid, test])
    if num_examples == 'all':
        return data
    return data[:num_examples]
