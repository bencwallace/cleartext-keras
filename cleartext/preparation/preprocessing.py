import re
import unicodedata

from tensorflow.keras.preprocessing import sequence


def clean(sentence):
    sentence = unicodedata.normalize('NFC', sentence)
    sentence = sentence.lower()

    # keep only alphanumeric and some punctuation
    pattern_keep = re.compile('A-Za-z0-9.!?,')
    sentence = pattern_keep.sub(' ', sentence)

    # add space around punctuation
    sentence = re.sub(r'([.!?,])', r' \1 ', sentence)

    return sentence


def prepare(df, tokenizer, seq_len='max'):
    # clean
    df = df.applymap(clean)

    # tokenize
    df = df.applymap(lambda s: f'<start> {s} <end>')
    tokenizer.fit_on_texts(df['source'])
    df = df.apply(lambda col: tokenizer.texts_to_sequences(col), axis=0)

    seq_len = seq_len if type(seq_len) is int else max(df.applymap(len).apply(max))
    df = df.applymap(lambda x: sequence.pad_sequences([x], padding='post', maxlen=seq_len)[0])
    return df
