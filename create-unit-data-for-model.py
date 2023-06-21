import json
from collections import defaultdict

import nlp2

data_dir = "./data/"

suffix = "ashraq_esc50"
for d in nlp2.get_files_from_dir(data_dir):
    if suffix not in d:
        continue

    print("filename", d)
    # ds = nlp2.read_json(d)
    ds = []
    with open(d) as f:
        for line in f:
            ds.append(json.loads(line))

    ## change output dir from data_dir to "model_data"
    result_filename = d.replace(data_dir, "./model_data/")
    fold_data = defaultdict(list)

    # create five fold data base on fold id
    for i in ds:
        code = i['merged_code']
        code_str = "".join(f"v_tok_{tok}" for tok in code)
        text = i['category']
        fold_id = i['fold']
        fold_data[fold_id].append([code_str, text])

    for fold_id in range(1, 6):
        # merge all folds except current fold to create training set
        train_set = []
        for f_id, data in fold_data.items():
            if f_id != fold_id:
                train_set += data

        # create pandas dataframe for training and validation sets
        train_data = train_set
        valid_data = fold_data[fold_id]

        nlp2.write_csv(train_data, result_filename.replace('.jsonl', f"_train_fold_{fold_id}.csv"))
        nlp2.write_csv(valid_data, result_filename.replace('.jsonl', f"_valid_fold_{fold_id}.csv"))

suffix = "superb_asr"
for d in nlp2.get_files_from_dir(data_dir):
    if suffix not in d:
        continue

    print("filename", d)
    # ds = nlp2.read_json(d)
    ds = []
    with open(d) as f:
        for line in f:
            ds.append(json.loads(line))

    ## change output dir from data_dir to "model_data"
    result_filename = d.replace(data_dir, "./model_data/")
    result_filename = result_filename.replace('.jsonl', ".csv")
    result = []
    for i in ds:
        code = i['merged_code']
        code_str = "".join(f"v_tok_{tok}" for tok in code)
        text = i['text'].lower()
        result.append([code_str, text])

    nlp2.write_csv(result, result_filename)

suffix = "superb_ks"
for d in nlp2.get_files_from_dir(data_dir):
    if suffix not in d:
        continue

    print("filename", d)
    # ds = nlp2.read_json(d)
    ds = []
    with open(d) as f:
        for line in f:
            ds.append(json.loads(line))

    ## change output dir from data_dir to "model_data"
    result_filename = d.replace(data_dir, "./model_data/")
    result_filename = result_filename.replace('.jsonl', ".csv")
    result = []
    for i in ds:
        code = i['merged_code']
        code_str = "".join(f"v_tok_{tok}" for tok in code)
        text = i['label']
        result.append([code_str, text])

    nlp2.write_csv(result, result_filename)
