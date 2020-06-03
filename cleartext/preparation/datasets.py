import io
import os

import dotenv
import pandas as pd


def load_wiki(dataset='wikismall', keep_splits=False):
    dotenv.load_dotenv()
    proj_dir = os.getenv('PROJECT_DIR')
    wiki_dir = os.path.join(proj_dir, 'data/raw/data-simplification', dataset)

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

    if keep_splits:
        return train, valid, test

    return pd.concat([train, valid, test])
