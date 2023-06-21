import argparse
import sys

import jsonlines
import torch
from asrp import extract_x_vector, extract_d_vector
from datasets import load_dataset

from speech2unit_model.hubert import hubert_layer9_code500, hubert_layer6_code50, hubert_layer6_code100, \
    hubert_layer6_code200
from speech2unit_model.mhubert import mhubert_layer11_code1000

ModelMap = {
    'hubert_layer9_code500': hubert_layer9_code500,
    'hubert_layer6_code50': hubert_layer6_code50,
    'hubert_layer6_code100': hubert_layer6_code100,
    'hubert_layer6_code200': hubert_layer6_code200,
    'mhubert_layer11_code1000': mhubert_layer11_code1000,
}


def jsonify(data):
    json_data = dict()
    for key, value in data.items():
        if isinstance(value, list):  # for lists
            value = [jsonify(item) if isinstance(item, dict) else item for item in value]
        if isinstance(value, dict):  # for nested lists
            value = jsonify(value)
        if isinstance(key, int):  # if key is integer: > to string
            key = str(key)
        if type(value).__module__ == 'numpy':  # if value is numpy.*: > to python list
            value = value.tolist()
        json_data[key] = value
    return json_data


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_sec", type=int, default=30, help="chunk sec, default 30")
    parser.add_argument("--feat_norm", action="store_true", help="normalize feature")
    parser.add_argument("--beamsearch", action="store_true", help="enable beamsearch")
    parser.add_argument("--topk", type=int, default=3, help="topk, default 3")
    parser.add_argument("--beamsize", type=int, default=1, help="beamsize, default 1")
    parser.add_argument("--ds", default='superb', type=str)
    parser.add_argument("--ds_split", default='asr', type=str)
    input_arg, model_arg = parser.parse_known_args(args)
    input_arg = {k: v for k, v in vars(input_arg).items() if v is not None}
    model_arg = {k.replace("--", ""): v for k, v in zip(model_arg[:-1:2], model_arg[1::2])}
    return input_arg, model_arg


def main(arg=None):
    input_arg, model_arg = parse_args(sys.argv[1:]) if arg is None else parse_args(arg)
    dataset = load_dataset(input_arg['ds'], input_arg['ds_split'])
    dataset.pop('train')
    hc_model = {k: v() for k, v in ModelMap.items()}

    def convert_code_fn(data):
        wav_tensor = torch.from_numpy(data['audio']['array'].astype('float32')).unsqueeze(0)
        sampling_rate = data['audio']['sampling_rate']
        for k, v in hc_model.items():
            hubert = v(input_values=wav_tensor,
                       feat_norm=False,
                       beamsearch=False,
                       top_k=3, beamsize=1)
            data.update({k + "_" + i: j for i, j in hubert.items()})
        data['x_vector'] = []
        data['d_vector'] = []
        try:
            data['x_vector'] = extract_x_vector(wav_tensor=wav_tensor, sampling_rate=sampling_rate)
        except:
            data['x_vector'] = []
            pass
        try:
            data['d_vector'] = extract_d_vector(wav_tensor=wav_tensor, sampling_rate=sampling_rate)
        except:
            data['d_vector'] = []
            pass
        return data

    new_ds = dataset.map(convert_code_fn)
    try:
        new_ds = new_ds.remove_columns(['file'])
    except:
        pass
    try:
        new_ds = new_ds.remove_columns(['filename'])
    except:
        pass
    new_ds = new_ds.remove_columns(['audio'])

    for k, v in new_ds.items():
        with jsonlines.open(
                f'./{input_arg["ds"].replace("/", "_")}_{input_arg["ds_split"]}_{k}_chunk_{input_arg["chunk_sec"]}_speaker.jsonl',
                mode='w') as writer:
            for d in v:
                writer.write(jsonify(d))


if __name__ == "__main__":
    main()
