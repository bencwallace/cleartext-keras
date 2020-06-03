def preprocess(sentence):
    sentence = re.sub(r'([?.!])', r' \1 ', sentence)
    sentence = re.sub('\s{2,}', ' ', sentence)
    sentence = sentence.strip()
    return f'<START> {sentence} <END>'


def pad(sequence):
    pass
