from metrics import BLEU, METEOR, ROUGE
import argparse
import logging
import os
import json
import torch
from transformers import AutoTokenizer
from typing import Dict

logger = logging.getLogger(__name__)

def evaluate(args, tokenizer) -> Dict:

    eval_output_dir = args.output_dir
    os.makedirs(eval_output_dir, exist_ok=True)

    ## generated text
    with open(f"{args.inference}", "r") as f:
        preds_file=json.load(f)

    ## ground truth
    with open(f"{args.labels}", "r") as f:
        refs_file=json.load(f)

    metrics = [
        BLEU(),
        ROUGE(),
        METEOR()
    ]

    for metric in metrics:
        for idx, dic in enumerate(preds_file):
            if refs_file[idx]["target"]==1:
                if dic["target"]==1:
                    metric.update((dic["response"], refs_file[idx]['response'])) ## hypothesis, reference 순서

    result = dict()
    output_eval_file = os.path.join(eval_output_dir, f"eval_results_{args.task}.txt")

    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results %s *****" % args.task)
        writer.write("***** Eval results %s *****\n" % args.task)
        for metric in metrics:
            name = metric.name()
            score = metric.compute()
            if metric.is_single:
                result[name] = score
                logger.info("  %s = %s", name, str(score))
                writer.write("%s = %s\n" % (name, str(score)))
            else:
                for _name, _score in zip(name, score):
                    result[_name] = _score
                    logger.info("  %s = %s", _name, str(_score))
                    writer.write("%s = %s\n" % (_name, str(_score)))

    return result


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--inference", type=str, default="val",
                        help="inferenced labels file name")
    parser.add_argument("--labels", type=str, default=None,
                        help="real label file name")
    parser.add_argument("--output_dir", type=str, default="", help="output dir")
    parser.add_argument("--task", type=str, default="", help="version 별 관리")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    args = parser.parse_args()


    # Setup CUDA & GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Evaluation
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    result = evaluate(args,tokenizer)

    return result


if __name__ == "__main__":
    main()
