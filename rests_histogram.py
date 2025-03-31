import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

all_rests = defaultdict(int)

for p in Path("./baroque_processed/").glob("**/*.json"):
    arr = np.array(json.loads(p.read_text()))[:, 1].astype(float)
    arr[1:] = arr[1:] - arr[:-1]
    assert arr.min() >= 0
    for k, v in Counter(map(lambda x: f"{x:.2f}", arr.tolist())).items():
        all_rests[k] += v

sorted_dict = dict(sorted(all_rests.items(), key=lambda x: x[1]))

Path("rests_histogram.json").write_text(json.dumps(sorted_dict))
