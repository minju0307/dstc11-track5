import json
import math
import os
import random
from itertools import islice
from multiprocessing import Process
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import joblib
import typer
from tqdm import tqdm
from transformers import T5Tokenizer, AutoTokenizer


DATA_FILE = 'data/pretrain_last_response.txt'
DATASET_CACHE_PATH = Path("pretrain_last_response/")
DATASET_CACHE_PATH.mkdir(exist_ok=True, parents=True)

dot_token: int
dot_token_1: int
mask_tokens: list


def write_disk(input_ids, target_ids, file_counter):
    print("New thread: writing file: " + str(DATASET_CACHE_PATH / (Path("dataset_" + str(file_counter)).stem + ".jbl")))
    joblib.dump([input_ids, target_ids],  # performance bottleneck 2 here. Now in separate process
                DATASET_CACHE_PATH / (Path("dataset_" + str(file_counter)).stem + ".jbl"))
    # open(CACHE_PATH / (Path("test").stem + ".jbl.example"), "w").write(str([input_ids, target_ids]))
    # print("\rFile written: " + str(CACHE_PATH / (Path("dataset_" + str(file_counter)).stem + ".jbl")))


def main(tokenizer_name: str = typer.Option("t5-base", help="T5 tokenizer used for token ids."),
         valid_size: float = typer.Option(0.1, help="Validation set size."),
         dumps_size: int = typer.Option(1000, help="Size in MB for the dataset raw files."),
         mask_probability: float = typer.Option(0.15, help="Probability of masking a token in a sentence.")):
    """This script preprocesses and tokenizes a standardized pretraining text Dataset (a file with a sentence in each
    line) into a set of tokenized files for training and validating the text2text model."""
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_name, model_max_length=2048)
    electra_tokenizer = AutoTokenizer.from_pretrained("google/electra-base-generator")

    global dot_token, dot_token_1, mask_tokens
    dot_token = tokenizer.convert_tokens_to_ids(["."])[0]
    dot_token_1 = tokenizer.convert_tokens_to_ids([")."])[0]
    mask_tokens = tokenizer.additional_special_tokens_ids
    meta = {}
    words_per_dump = 3000000 * dumps_size  # approx. 300_000 words per mb of dump file.

    error_count=0

    with open(DATA_FILE, 'r') as in_file:
        number_lines = len([0 for _ in in_file])
        in_file.seek(0)  # after reading number of lines, restart file pointer
        n = 1000  # size of batches of sentences from input file. ~=100mb chunks
        batch_counter, file_counter, words_counter = 1, 1, 0
        input_ids, target_ids = [], []

        for sentence_batch in iter(lambda: tuple(islice(in_file, n)), ()):  # tuples of islices size n until tuple ()
            print(f"Processing batch {batch_counter} of {math.ceil(number_lines / n)}.")

            for line in sentence_batch:
                turn_list=line.split('\t')
                line_input_text=' '.join(turn_list[:-1])+' <extra_id_0>'
                line_target_text='<extra_id_0>'+turn_list[-1]

                line_input_ids = tokenizer.encode(line_input_text, return_attention_mask=False, verbose=False)
                line_target_ids = tokenizer.encode(line_target_text, return_attention_mask=False, verbose=False)

                ## 전체에 넣어주기
                input_ids.append(line_input_ids)
                target_ids.append(line_target_ids) #target에 eos 토큰 추가하기

            ## 잘들어갔는지 확인
            print(f'input_test: {tokenizer.batch_decode(input_ids)[0]}')
            print(f'target_test: {tokenizer.batch_decode(target_ids)[0]}')
            print()

            for x in input_ids: words_counter +=len(x)

            if words_counter > words_per_dump:  # 30M words ~= 100MB dump file size
                dump_size = int(len(input_ids) * words_per_dump / words_counter)
                meta[f"dataset_{file_counter}.jbl"] = dump_size
                Process(target=write_disk, args=(input_ids[:dump_size], target_ids[:dump_size], file_counter)).start()
                input_ids, target_ids = input_ids[dump_size:], target_ids[dump_size:]
                file_counter += 1
                words_counter -= words_per_dump
            batch_counter += 1
        Process(target=write_disk, args=(input_ids, target_ids, file_counter)).start()  # write last dump to disk
        meta[f"dataset_{str(file_counter)}.jbl"] = len(input_ids)

    print("Dataset tokenized. Partitioning...")
    total_size = sum(meta.values())
    train, valid = [], []
    count, train_size = 0, 0
    for file, size in meta.items():
        count += size
        if count < (1 - valid_size) * total_size:
            train_size += size
            train.append(file)
        else:
            valid.append(file)
    meta["train_size"], meta["valid_size"] = train_size, total_size - train_size
    meta["train"], meta["valid"] = train, valid
    with open(DATASET_CACHE_PATH / Path("dataset_meta.json"), 'w') as json_file:
        json.dump(meta, json_file, indent=2)
    print("Dataset ready. Meta file written to " + str(DATASET_CACHE_PATH / Path("dataset_meta.json")))


if __name__ == "__main__":
    typer.run(main)
