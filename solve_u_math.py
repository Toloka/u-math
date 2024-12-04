import argparse
import json

from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm

from prompts import solve_cot_prompt


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Solve U-Math problems using CoT prompt."
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://api.openai.com/v1",
        help="Base url for OpenAI-compatible endpoint.",
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
        default="predictions_u_math.json",
        help="Output JSON file.",
    )
    args = parser.parse_args()

    # Load the dataset
    dataset = load_dataset("toloka/u-math")

    # Make openai client
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    # Predict with CoT prompt
    predictions = {}
    for item in tqdm(dataset):
        prompt = solve_cot_prompt(
            problem_statement=item["problem_statement"],
            image=item["image"],
        )
        response = client.chat.completions.create(
            messages=prompt,
            max_tokens=4096,
            temperature=0.0,
            model=args.model,
        )
        predictions[item["uuid"]] = response.choices[0].message

    # Save predictions to JSON file
    with open(args.output_file, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to {args.output_file} as uuid -> prediction json.")


if __name__ == "__main__":
    main()
