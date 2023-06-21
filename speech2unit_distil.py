import argparse
import sys

import jsonlines
import torch
from datasets import load_dataset, Audio
from transformers import AutoProcessor, AutoModelForCTC


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model name")
    parser.add_argument("--ds", default='superb', type=str)
    parser.add_argument("--ds_split", default='asr', type=str)
    input_arg, model_arg = parser.parse_known_args(args)
    input_arg = {k: v for k, v in vars(input_arg).items() if v is not None}
    model_arg = {k.replace("--", ""): v for k, v in zip(model_arg[:-1:2], model_arg[1::2])}
    return input_arg, model_arg


def main(arg=None):
    input_arg, model_arg = parse_args(sys.argv[1:]) if arg is None else parse_args(arg)
    dataset = load_dataset(input_arg['ds'], input_arg['ds_split'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoProcessor.from_pretrained(input_arg['model'])
    model = AutoModelForCTC.from_pretrained(input_arg['model']).to(device)

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    def convert_code_fn(data):
        input_values = processor(data['audio']['array'], sampling_rate=data['audio']['sampling_rate'],
                                 return_tensors="pt",
                                 padding="longest").input_values
        logits = model(input_values.to(device)).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        data['merged_code'] = processor.tokenizer.tokenize(transcription[0])
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
        json_items = []
        for d in v:
            json_items.append(d)

        with jsonlines.open(
                f'./{input_arg["ds"].replace("/", "_")}_{input_arg["ds_split"]}_{k}_{input_arg["model"].replace("/", "_")}.jsonl',
                mode='w') as writer:
            writer.write_all(json_items)


if __name__ == "__main__":
    main()
