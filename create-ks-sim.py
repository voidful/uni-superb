from collections import defaultdict
from itertools import combinations  # , combinations_with_replacement

import editdistance as ed
import nlp2

data_dir = "./data/"
suffix = "superb_ks"

for d in nlp2.get_files_from_dir(data_dir):
    if suffix not in d or 'train' in d:
        continue

    print("filename", d)
    label_data = defaultdict(list)
    ds = nlp2.read_json(d)

    for d in ds:
        label = d['label']
        label_data[label].append(d["merged_code"])

    for k, v in label_data.items():
        print(k)
        comb = list(combinations(v, 2))
        for c in comb:
            print(c)
            print(ed.eval(*c))
            break
        print(len(comb))
    break
