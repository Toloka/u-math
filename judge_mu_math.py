import argparse
import json
import re

from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm

from prompts import judge_cot_prompt, judge_extract_prompt


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
            judge_cot = judge_response.choices[0].message["content"]
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
            extracted_judgment = extract_response.choices[0].message["content"]
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
            "extracted_judgment_binary": 1 if extracted_judgment_clean == "yes" else 0,
            "correct_judgment_label": item["label"],
        }

    # Save judgments to JSON file
    with open(args.output_file, "w") as f:
        json.dump(judgments, f, indent=2)
    print(f"Judgments saved to {args.output_file} as uuid -> judgments JSON.")

    # Print final accuracy of the judgments: total, per model split
    # TODO: print scores



if __name__ == "__main__":
    main()
