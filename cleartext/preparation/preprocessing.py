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

    return f'<START> {sentence} <END>'


def prepare(df, tokenizer, pad=True):
    # clean
    df = df.applymap(clean)

    # tokenize
    tokenizer.fit_on_texts(df['source'])
    df = df.apply(lambda col: tokenizer.texts_to_sequences(col), axis=0)

    # pad
    max_len = None
    if pad:
        max_len = max(df.applymap(len).max(axis=0))
        df = df.applymap(lambda x: sequence.pad_sequences([x], maxlen=max_len, padding='post')[0])

    return df, max_len
