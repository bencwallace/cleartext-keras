import io
import os

import tensorflow as tf


github_url = 'https://raw.githubusercontent.com/louismartin/dress-data/master/data-simplification.tar.bz2'
_, file_name = os.path.split(github_url)

data_dir = '../../data'


def load_wiki(dataset='wikismall'):
    zip_path = tf.keras.utils.get_file(file_name, github_url, extract=True, cache_dir='../../')
    zip_dir = os.path.dirname(zip_path)
    wiki_dir = os.path.join(zip_dir, 'data-simplification', dataset)

    prefix = 'PWKP_108016.tag.80.aner.ori' if dataset == 'wikismall' else 'wiki.full.aner.ori'
    file_path = '.'.join([prefix, 'test', 'src'])
    io.open(os.path.join(wiki_dir, file_path))

    data = []
    for split in ['train', 'valid', 'test']:
        for loc in ['src', 'dst']:
            file_name = '.'.join([prefix, split, loc])
            file_path = os.path.join(wiki_dir, file_name)
            stream = io.open(file_path)
            lines = stream.read().split('\n')
            data.append(lines)

    # returns src_train, dst_train, src_valid, dst_valid, src_test, dst_test
    return data
