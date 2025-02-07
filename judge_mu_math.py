import argparse
import json
import re

from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm

from collections import defaultdict
from itertools import chain
from operator import itemgetter

from functools import partial
from prompts import judge_cot_prompt, judge_extract_prompt


def cons(x, xs):
    return chain((x,), xs)


def eager_groupby(xs, key=None):
    d = defaultdict(list)
    for x in xs:
        d[x if key is None else key(x)].append(x)
    return d


def scores(tp, tn, fp, fn):
    tpr = tp / (tp + fn) if tp + fn > 0 else 0
    tnr = tn / (tn + fp) if tn + fp > 0 else 0
    ppv = tp / (tp + fp) if tp + fp > 0 else 0
    npv = tn / (tn + fn) if tn + fn > 0 else 0
    f1 = (tp / (2*tp + fp + fn) if tp + fp + fn > 0 else 1/2) + (tn / (2*tn + fp + fn) if tn + fp + fn > 0 else 1/2)
    return f1, tpr, tnr, ppv, npv


def stats(matches):
    # match == (y_true, y_pred)
    aggs = eager_groupby(matches)
    return (len(aggs[(True, True)]), len(aggs[(False, False)]), len(aggs[(False, True)]), len(aggs[(True, False)]))


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Judge U-Math problem solutions using CoT and Extract prompts."
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://api.openai.com/v1",
        help="Base URL for OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="stub",
        help="Your API key for OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name for OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="judgments_u_math.json",
        help="Output JSON file with judgments.",
    )
    args = parser.parse_args()

    # Load the dataset
    dataset = load_dataset("toloka/mu-math", split="test")

    # Make OpenAI client
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    # Judge the predictions using CoT and Extract prompts
    judgments = {}
    for item in tqdm(dataset):
        # Generate the judge CoT prompt
        judge_prompt = judge_cot_prompt(
            problem_statement=item["problem_statement"],
            golden_answer=item["golden_answer"],
            generated_answer=item["model_output"],
        )
        judge_response = client.chat.completions.create(
            messages=judge_prompt,
            max_tokens=4096,
            temperature=0.0,
            model=args.model,
        )
        try:
            judge_cot = judge_response.choices[0].message.content
        except:
            print(f"Error with UUID: {item['uuid']}")
            judge_cot = ""

        # Generate the judge Extract prompt
        extract_prompt = judge_extract_prompt(generated_judgment=judge_cot)
        extract_response = client.chat.completions.create(
            messages=extract_prompt,
            max_tokens=10,
            temperature=0.0,
            model=args.model,
        )
        try:
            extracted_judgment = extract_response.choices[0].message.content
        except:
            print(f"Error with UUID: {item['uuid']}")
            extracted_judgment = ""

        # Clean the extracted judgment: remove \boxed, \text and .
        extracted_judgment_clean = re.sub(
            r"\\(boxed|text)\s*", "", extracted_judgment.lower()
        )
        extracted_judgment_clean = re.sub(r"[{}.]", "", extracted_judgment_clean)

        # Store the judgments
        judgments[item["uuid"]] = {
            "judge_cot": judge_cot,
            "extracted_judgment": extracted_judgment,
            "extracted_judgment_binary": (
                extracted_judgment_clean.endswith("yes")
                if (extracted_judgment_clean.endswith("yes") or extracted_judgment_clean.endswith("no"))
                else (not item["label"])
            ),
            "correct_judgment_label": item["label"],
        }

    # Save judgments to JSON file
    with open(args.output_file, "w") as f:
        json.dump(judgments, f, indent=2)
    print(f"Judgments saved to {args.output_file} as uuid -> judgments JSON.")

    # convert dataset to dict: {uuid: item}
    dataset_dict = {item["uuid"]: item for item in dataset}

    # Print final scores of the judgments: total, per-model split
    jks, jvs = zip(*judgments.items())
    models = [dataset_dict[uuid]["model"] for uuid in jks]
    matches = [(record["correct_judgment_label"], record["extracted_judgment_binary"]) for record in jvs]
    splits, splitmatches = zip(*eager_groupby(zip(models, matches), key=itemgetter(0)).items())

    print("scores: macro-F1 / TPR / TNR / PPV / NPV, %")
    splitstats = [*map(stats, map(partial(map, itemgetter(1)), splitmatches))]
    for s, st in zip(cons(None, splits), cons(map(sum, zip(*splitstats)), splitstats)):
        print("mu-MATH" + (f" {s} " if s else " ") + "scores: " + " / ".join(f"{x*100:.1f}" for x in scores(*st)))


if __name__ == "__main__":
    main()
