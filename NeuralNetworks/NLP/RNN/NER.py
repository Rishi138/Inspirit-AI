import pandas as pd
import keras

df = pd.read_csv('ner_dataset.csv', encoding= 'unicode_escape')
print(df.info())
print(df.head())


def get_dict_map(data, token_or_tag):
    tok2idx = {}
    idx2tok = {}

    if token_or_tag == 'token':
        vocab = list(set(data['Word'].to_list()))
    else:
        vocab = list(set(data['Tag'].to_list()))

    idx2tok = {idx: tok for idx, tok in enumerate(vocab)}
    tok2idx = {tok: idx for idx, tok in enumerate(vocab)}
    return tok2idx, idx2tok


token2idx, idx2token = get_dict_map(df, 'token')
tag2idx, idx2tag = get_dict_map(df, 'tag')

