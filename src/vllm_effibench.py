import argparse
import os
import json
from tqdm import tqdm
import copy
import openai
import sys
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from datasets import load_dataset
import time
from vllm import LLM, SamplingParams


from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

batch_size = 64
checkpoints = ["EffiCoder/Qwen2.5-7B-lr5e-6-epoch4-final","EffiCoder/deepseek-coder-6.7b-base-lr5e-6-epoch4-final","EffiCoder/deepseek-coder-6.7b-ins-lr5e-6-epoch4-final","EffiCoder/CodeLlama-7b-hf-lr5e-6-epoch4-final","meta-llama/CodeLlama-7b-hf","Qwen/Qwen2.5-7B","deepseek-ai/deepseek-coder-6.7b-base","deepseek-ai/deepseek-coder-6.7b-instruct"]
# "deepseek-ai/deepseek-coder-1.3b-instruct","deepseek-ai/deepseek-coder-6.7b-instruct","deepseek-ai/deepseek-coder-33b-instruct","EffiCoder/deepseek-coder-1.3b-instruct-new","EffiCoder/deepseek-coder-6.7b-instruct-new"
def construct_prompt_template(inputs, llm, sampling_params):
    outputs = llm.generate(inputs, sampling_params)
    generated_texts = []
    for i in range(len(outputs)):
        generated_texts.append(outputs[i].outputs[0].text)
    return generated_texts


def fetch_completion(data_entry_lists, llm, sampling_params):
    inputs_batchs = []
    for data_entry in data_entry_lists:
        # inputs_batchs.append(f"Please complete Python code based on the task description and test cases. # Task description:\n```python\n{data_entry['markdown_description']}\n{data_entry['small_test_cases']}\n```\n#Solution:\n")
#         inputs_batchs.append(f'''
# Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
# ```python
# {data_entry['markdown_description']}
# {data_entry['small_test_cases']}
# ```
# ''')

        inputs_batchs.append(f'''
Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
```python
{data_entry['prompt']}
```
''')
    completion_lists = construct_prompt_template(inputs_batchs, llm, sampling_params)
    for i in range(len(data_entry_lists)):
        data_entry_lists[i]["completion"] = completion_lists[i]

    return data_entry_lists

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--checkpoint",
        type=str,
        default="m-a-p/OpenCodeInterpreter-DS-1.3B",
        required=True,
    )
    args = args.parse_args()
    checkpoint = args.checkpoint
    with open("../datasets/inference.json", "r") as f:
        dataset = json.load(f)
    print(checkpoint)
    
    llm = LLM(
        model=checkpoint,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        enable_prefix_caching=True,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=0, top_p=1, max_tokens=1024, stop=[tokenizer.eos_token]
    )

    for i in tqdm(range(0,len(dataset),batch_size)):
        if i+batch_size > len(dataset):
            dataset[i:] = fetch_completion(dataset[i:], llm, sampling_params)
        else:
            dataset[i:i+batch_size] = fetch_completion(dataset[i:i+batch_size], llm, sampling_params)

    end_name = checkpoint.split("/")[-1]
    # with open(f"../../results/leetcode_{end_name}_lr1e-5-epoch2_0.json", "w") as f:
    #     json.dump(dataset, f, indent=4)
    with open(f"../../results/leetcode_{end_name}_0.json", "w") as f:
        json.dump(dataset, f, indent=4)