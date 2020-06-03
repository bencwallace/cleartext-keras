import re
import unicodedata

from tensorflow.keras.preprocessing import sequence


def preprocess(sentence):
    sentence = unicodedata.normalize('NFC', sentence)
    sentece = sentence.lower()
    sentence = re.sub(r'([?.!])', r' \1 ', sentence)
    sentence = re.sub('\s{2,}', ' ', sentence)
    sentence = sentence.strip()
    return f'<START> {sentence} <END>'


def prepare(df, tokenizer, pad=True):
    # preprocess
    df = df.applymap(preprocess)

    # tokenize
    tokenizer.fit_on_texts(df['source'])
    df = df.apply(lambda col: tokenizer.texts_to_sequences(col), axis=0)

    # pad
    if pad:
        max_len = max(df.applymap(len).max(axis=0))
        df = df.applymap(lambda x: sequence.pad_sequences([x], maxlen=max_len, padding='post')[0])

    return df
