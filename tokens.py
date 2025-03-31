from torch import Tensor


tokens = {
    'PITCH_-12': 0,
    'PITCH_-11': 1,
    'PITCH_-10': 2,
    'PITCH_-9': 3,
    'PITCH_-8': 4,
    'PITCH_-7': 5,
    'PITCH_-6': 6,
    'PITCH_-5': 7,
    'PITCH_-4': 8,
    'PITCH_-3': 9,
    'PITCH_-2': 10,
    'PITCH_-1': 11,
    'PITCH_0': 12,
    'PITCH_1': 13,
    'PITCH_2': 14,
    'PITCH_3': 15,
    'PITCH_4': 16,
    'PITCH_5': 17,
    'PITCH_6': 18,
    'PITCH_7': 19,
    'PITCH_8': 20,
    'PITCH_9': 21,
    'PITCH_10': 22,
    'PITCH_11': 23,
    'PITCH_12': 24,
    'REST_1': 25,
    'REST_2': 26,
    'REST_3': 27,
    'REST_4': 28,
    'REST_5': 29,
    'REST_6': 30,
    'REST_7': 31,
    'REST_8': 32,
    'REST_9': 33,
    'REST_10': 34,
    'REST_11': 35,
    'REST_12': 36,
    'PLAY_0': 37,
    'PAD_0': 38,
}

antitokens = {v: k for k, v in tokens.items()}


def encode(seq):
    return [tokens[s] for s in seq]


def decode(seq):
    if isinstance(seq, Tensor):
        return [antitokens[s] for s in seq.tolist()]
    return [antitokens[s] for s in seq]
