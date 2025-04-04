import json
from collections import Counter, defaultdict
from pathlib import Path
import sympy

import numpy as np

all_rests = defaultdict(int)

for p in Path("./baroque_processed/").glob("**/*.json"):
    arr = np.array(json.loads(p.read_text()))[:, 1].astype(float)
    arr[1:] = arr[1:] - arr[:-1]
    assert arr.min() >= 0
    for k, v in Counter(map(lambda x: round(100 * x), arr.tolist())).items():
        all_rests[k] += v

all_rests = {int(k): v for k, v in all_rests.items()}

sorted_dict = dict(sorted(all_rests.items(), key=lambda x: x[1]))

Path("rests_histogram.json").write_text(json.dumps(sorted_dict))


divisiors_ranking = defaultdict(int)
for k, v in all_rests.items():
    for d in list(sympy.divisors(k))[-3:]:
        divisiors_ranking[d] += v

divisiors_ranking = dict(sorted(divisiors_ranking.items(), key=lambda x: x[1], reverse=True))

chosen_ids = range(0, 16)

divisiors_ranking_items = list(divisiors_ranking.items())
chosen = dict(divisiors_ranking_items[i] for i in chosen_ids)

Path("divisors.json").write_text(json.dumps(divisiors_ranking))
