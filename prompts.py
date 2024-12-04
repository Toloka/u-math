from typing import Any


def _get_openai_content_with_image(
    text: str,
    image: str | None = None,
) -> str | list[dict[str, Any]]:
    """
    Get OpenAI message content with image if present
    :param text: Text message
    :param image: Optional image in base64 or URL format
    :return: OpenAI message content
    """
    if image:
        return [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image}},
        ]
    else:
        return text


def solve_cot_prompt(
    problem_statement: str,
    image: str | None = None,
) -> list[dict]:
    """
    Produce "Solve using CoT" prompt from
    https://github.com/deepseek-ai/DeepSeek-Math/blob/main/README.md
    :param problem_statement: problem text
    :param image: base64 encoded image if present
    :return: Prompt formatted for the OpenAI API.
    """
    deepseek_prompt = (
        "Please reason step by step, and put your final answer within \\boxed{}"
    )
    return [
        {
            "role": "user",
            "content": _get_openai_content_with_image(
                problem_statement + "\n" + deepseek_prompt, image
            ),
        },
    ]


def judge_cot_prompt(
    problem_statement: str,
    golden_answer: str,
    generated_answer: str,
) -> list[dict]:
    """
    Produce "Judge with CoT" Prompt
    :param problem_statement: The text of the problem to be graded.
    :param golden_answer: The correct reference answer for the problem.
    :param generated_answer: The student's answer to be evaluated.
    :return: Prompt formatted for the OpenAI API.
    """
    prompt = """You'll be provided with a math problem, a correct answer for it and a solution for evaluation.
You have to answer whether the solution is correct or not.
---
PROBLEM STATEMENT:
{}

CORRECT ANSWER:
{}

SOLUTION TO EVALUATE:
{}
---
Now please compare the answer obtained in the solution with the provided correct answer to evaluate whether the solution is correct or not.

Think step-by-step, following these steps, don't skip any:
1. Extract the answer from the provided solution
2. Make any derivations or transformations that may be necessary to compare the provided correct answer with the extracted answer
3. Perform the comparison
4. Conclude with your final verdict — put either "Yes" or "No" on a separate line"""

    return [
        {
            "role": "user",
            "content": prompt.format(
                problem_statement, golden_answer, generated_answer
            ),
        }
    ]


def judge_extract_prompt(
    generated_judgment: str,
) -> list[dict]:
    """
    Produce "Extract judgment" prompt
    :param generated_judgment: The text of the judgment to be extracted.
    :return:  Prompt formatted for the OpenAI API.
    """
    prompt = """You'll be given a result of an evaluation of some mathematical solution by a professional evaluator.
You need to extract the final verdict of this evaluation in simple terms: is the solution graded as correct or not.
Output only a single label — "Yes", "No" or "Inconclusive" — according to the provided evaluation ("Yes" if the solution is graded as correct, "No" if the solution is graded as incorrect, "Inconclusive" if the evaluation is incomplete or the final verdict is not settled upon).
Only output "Inconclusive" for incomplete or unsettled evaluations. If the evaluation does contain a single final verdict like "Yes", "Correct", "True", "No", "Incorrect", "False" and so on, even if it is supplied with some additional disclaimers and remarks, output a "Yes" or "No" label accordingly. 

Here goes your input:
```
{}
```

Now please output exactly either "Yes", "No" or "Inconclusive"."""

    return [{"role": "user", "content": prompt.format(generated_judgment)}]
