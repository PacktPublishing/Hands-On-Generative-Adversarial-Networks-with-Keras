from collections import defaultdict, Counter
import numpy as np
from glob import glob
import pandas as pd


def iterate_minibatches(data, token_to_id, vocabulary_size, max_sentence_length,
                        batch_size=128):
    one_hot_matrix = np.eye(vocabulary_size)
    i = 0
    while True:
        # drop last?
        if len(data) - i*batch_size < batch_size:
            ids = np.random.randint(0, len(data), len(data))
            np.random.shuffle(ids)
            i = 0

        if i == 0:
            ids = np.random.randint(0, len(data), len(data))
            np.random.shuffle(ids)

        train_data = np.array([
            sentence_to_onehot(data[k], token_to_id, one_hot_matrix, max_sentence_length)
            for k in ids[i*batch_size:(i+1)*batch_size]])

        i = (i + 1) % len(data)
        yield train_data


def sentence_to_onehot(sentence, token_to_id, one_hot_matrix,
                       max_sentence_length):
    ids = [token_to_id[token] for token in sentence]
    # 'pad' to max_max_sentence_length
    if len(ids) < max_sentence_length:
        ids.extend([ids[-1]] * (max_sentence_length - len(ids)))

    # crop to max_max_sentence_length
    ids = ids[:max_sentence_length]
    return one_hot_matrix[ids][None, :]


def sample(generator_output, id_to_token, join_char=' '):
    ids_per_batch = np.argmax(generator_output, axis=3)
    sentences = []
    for item in ids_per_batch:
        sentences.append(join_char.join([id_to_token[id] for id in item[0]]))
    return sentences


def print_data(sentences):
    for sentence in sentences:
        print("-->", sentence)


def get_data(glob_str, vocabulary_size=96, sos="<s>", eos="</s>", unk_token='|',
             symbol_based=True):

    print("Loading data")
    data = None
    i = 0
    for filepath in glob(glob_str):
        with open(filepath) as f:
            cur_data = f.readlines()
            if symbol_based:
                cur_data = [list(sos) + list(d.strip().lower()) + list(eos)
                            for d in cur_data]
            else:
                cur_data = [[sos] + d.strip().split() + [eos]
                            for d in cur_data]
            data = cur_data if data is None else data + cur_data
            if len(data) > 500000:
                print("number of sentences: ", len(data))
                break

    token_frequency = Counter(np.hstack(data))
    token_frequency = pd.DataFrame.from_dict(
        token_frequency, orient='index', columns=['frequency'])
    token_frequency = token_frequency.sort_values("frequency", ascending=False)

    unk_id = np.argwhere(token_frequency.index == unk_token)
    if len(unk_id) > 0:
        unk_id = unk_id[0, 0]
    else:
        unk_id = vocabulary_size - 1
        vocabulary_size -= 1

    token_frequency = token_frequency.drop(
        token_frequency.index[np.arange(vocabulary_size, len(token_frequency))])

    token_to_id = defaultdict(lambda: unk_id)
    id_to_token = defaultdict(lambda: unk_token)
    token_to_id[unk_token] = unk_id
    id_to_token[unk_id] = unk_token

    for i, token in enumerate(token_frequency.index):
        token_to_id[token] = i
        id_to_token[i] = token

    return data, token_to_id, id_to_token


def log_text(texts, name, i, logger):
    texts = '\n'.join(texts)
    logger.add_text('{}_{}'.format(name, i), texts, i)
