from torch import Tensor


antitokens = dict(enumerate([
    'PITCH_-12',
    'PITCH_-11',
    'PITCH_-10',
    'PITCH_-9',
    'PITCH_-8',
    'PITCH_-7',
    'PITCH_-6',
    'PITCH_-5',
    'PITCH_-4',
    'PITCH_-3',
    'PITCH_-2',
    'PITCH_-1',
    'PITCH_0',
    'PITCH_1',
    'PITCH_2',
    'PITCH_3',
    'PITCH_4',
    'PITCH_5',
    'PITCH_6',
    'PITCH_7',
    'PITCH_8',
    'PITCH_9',
    'PITCH_10',
    'PITCH_11',
    'PITCH_12',
    'REST_1',
    'REST_2',
    'REST_3',
    'REST_5',
    'REST_7',
    'REST_11',
    'REST_13',
    'REST_19',
    'REST_23',
    'REST_26',
    'REST_35',
    'REST_50',
    'REST_100',
    'REST_110',
    'REST_150',
    'PLAY_0',
    'PAD_0',
]))


tokens = {v: k for k, v in antitokens.items()}


def encode(seq):
    return [tokens[s] for s in seq]


def decode(seq):
    if isinstance(seq, Tensor):
        return [antitokens[s] for s in seq.tolist()]
    return [antitokens[s] for s in seq]
