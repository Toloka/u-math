# U-MATH and $\mu$-MATH evaluation code

This repository contains the official evaluation code for the U-MATH and $\mu$-MATH benchmarks. These datasets are designed to test the mathematical reasoning and meta-evaluation capabilities of Large Language Models (LLMs) on university-level problems.

### Overview

U-MATH provides a set of 1,100 university-level mathematical problems, while µ-MATH complements it with a meta-evaluation framework focusing on solution judgment with 1084 LLM solutions. 

* 📊 [U-MATH benchmark at Huggingface](https://huggingface.co/datasets/toloka/umath)
* 🔎 [μ-MATH benchmark at Huggingface](https://huggingface.co/datasets/toloka/mumath)
* 🗞️ [Paper](https://arxiv.org/abs/2412.03205)
* 👾 [Evaluation Code at GitHub](https://github.com/Toloka/u-math/)

### U-MATH Evaluation Results

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/650238063e61bc019201e3e2/beMyOikpKfp3My2vu5Mjc.png" alt="umath-table" width="800"/>
</div>

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/650238063e61bc019201e3e2/7_VZXidxMHG7PiDM983lS.png" alt="umath-bar" width="950"/>
</div>


### $\mu$-MATH Evaluation Results

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/650238063e61bc019201e3e2/lz_ylYOUd6BSK8yFn3K77.png" alt="mumath-table" width="1000"/>
</div>

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/650238063e61bc019201e3e2/Ook-JXum03E0UdWBIW0qB.png" alt="mumath-scatter" width="800"/>
</div>


### Structure and Usage

This repository provides scripts for solving and evaluating the U-MATH and μ-MATH datasets.

File Structure
* `solve_u_math.py`: Script to generate solutions for U-MATH problems using an OpenAI-compatible endpoint (e.g. gpt-4o or VLLM).
* `judge_u_math.py`: Script to evaluate the correctness of U-MATH solutions.
* `judge_mu_math.py`: Script to evaluate the quality of LLM judgments for μ-MATH solutions.
* `README.md`: This file.
* `requirements.txt`: List of dependencies required for running the scripts.


Download the repository and install the dependencies:
```shell
git clone https://github.com/toloka/u-math.git
cd u-math
pip install -r requirements.txt
```

#### Solve U-MATH Problems

To generate solutions for U-MATH problems, run the following command:
```shell
python solve_u_math.py --base_url <BASE_URL> --api_key <YOUR_API_KEY> --model <MODEL_NAME> --output_file predictions_u_math.json
```

#### Judge U-MATH Solutions

To evaluate the correctness of U-MATH solutions, run the following command:
```shell
python judge_u_math.py --base_url <BASE_URL> --api_key <YOUR_API_KEY> --model <MODEL_NAME> --predictions_file predictions_u_math.json --output_file judgments_u_math.json
```

#### Evaluate Judge on μ-MATH

To evaluate the quality of LLM judgments for μ-MATH solutions, run the following command:
```shell
python judge_u_math.py --base_url <BASE_URL> --api_key <YOUR_API_KEY> --model <MODEL_NAME> --output_file judgments_mu_math.json
```

### Licensing Information
* The contents of the μ-MATH's machine-generated `model_output` column are subject to the underlying LLMs' licensing terms.
* Contents of all the other dataset U-MATH and μ-MATH fields, as well as the code, are available under the MIT license.

### Citation
If you use U-MATH or μ-MATH in your research, please cite the paper:
```bibtex
@inproceedings{umath2024,
title={U-MATH: A University-Level Benchmark for Evaluating Mathematical Skills in LLMs},
author={Konstantin Chernyshev, Vitaliy Polshkov, Ekaterina Artemova, Alex Myasnikov, Vlad Stepanov, Alexei Miasnikov and Sergei Tilga},
year={2024}
}
```

### Contact
For inquiries, please contact kchernyshev@toloka.ai
