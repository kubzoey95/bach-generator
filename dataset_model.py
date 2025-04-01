from torch.utils.data import Dataset
from pathlib import Path
import json
import numpy as np
import random
from tokens import encode
import math
from functools import lru_cache
import torch


@lru_cache(4096)
def split(time, rests):
    out = []
    for r in sorted(rests, reverse=True):
        if time == 0:
            break
        if time < r:
            continue
        while time >= r:
            out.append(r)
            time -= r
    return out


def split_and_shuffle(time, rests):
    out = split(time, rests)
    random.shuffle(out)
    return out


class MusicDataset(Dataset):
    def __init__(self, dataset_dir, context_length=256):
        super().__init__()
        self.context_length = context_length
        self.pieces = [json.loads(p.read_bytes()) for p in Path(dataset_dir).glob("**/*.json")]
        self.pointers = [(i, j) for i, p in enumerate(self.pieces) for j in range(len(p))]

        self.rests_ms = (1, 2, 3, 5, 7, 11, 13, 19, 23, 26, 35, 50, 100, 150)
        self.pitches = (1,2,3,4,5,6,7,8,9,10,11,12)

    def __len__(self):
        return len(self.pointers)

    def load_sample(self, i, j):
        notes: list = self.pieces[i][max(j-self.context_length, 0):j+2]
        notes.sort(key=lambda x: (int(x[0] == "END"), x[1], random.randint(-99999, 99999)))
        preout = np.array([[n0[0], min(n1[1] - n0[1], 10)] for n0, n1 in zip(notes[:-1], notes[1:])])
        preout[:, 1] = (preout[:, 1] * 1000).round().astype(int)  # to miliseconds
        
        preout[1:, 0] = preout[1:, 0] - preout[:-1, 0]
        preout[0, 0] = preout[0, 0] + random.randint(-11, 11)

        out = []

        for pitch, time in preout:
            if pitch == 0:
                out.append(("PITCH", 0))
            else:
                for r in split_and_shuffle(abs(pitch), self.pitches):
                    out.append(("PITCH", np.sign(pitch) * r))
            out.append(("PLAY", 0))
            
            if time > 0:
                for r in split_and_shuffle(time, self.rests_ms):
                    out.append(("REST", r))

        out = [*(("PAD", 0) for _ in range(self.context_length + 1 - len(out))), *out]

        out = encode([f"{o[0]}_{int(o[1])}" for o in out])

        assert len(out) >= 257
        if len(out) > 257:
            i = random.randint(0, len(out) - 257)
            x = out[i:i+256]
            y = out[i+256]
        else:
            x = out[:256]
            y = out[256]
        
        assert len(x) == 256
        return torch.tensor(x), torch.tensor(y)

    def __getitem__(self, index):
        sample = self.load_sample(*self.pointers[index])
        return sample
