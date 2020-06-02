import io
import os


def load_wiki(dataset='wikismall', data_dir='../../data', keep_splits=False):
    wiki_dir = os.path.join(data_dir, 'raw/data-simplification', dataset)

    prefix = 'PWKP_108016.tag.80.aner.ori' if dataset == 'wikismall' else 'wiki.full.aner.ori'
    data = []
    for split in ['train', 'valid', 'test']:
        for loc in ['src', 'dst']:
            file_name = '.'.join([prefix, split, loc])
            file_path = os.path.join(wiki_dir, file_name)
            stream = io.open(file_path)
            lines = stream.read().split('\n')
            data.append(lines)

    if keep_splits:
        return data

    src_train, dst_train, src_valid, dst_valid, src_test, dst_test = data
    src = src_train + src_valid + src_test
    dst = dst_train + dst_valid + dst_test
    return src, dst
