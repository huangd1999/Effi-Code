from transformers import AutoTokenizer
import json
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
import argparse

batch_size = 64
# "EffiCoder/CodeLlama-7b-hf-lr5e-6-epoch4-final","meta-llama/CodeLlama-7b-hf","Qwen/Qwen2.5-7B","deepseek-ai/deepseek-coder-6.7b-base",
# "EffiCoder/Qwen2.5-7B-lr5e-6-epoch4-final","EffiCoder/deepseek-coder-6.7b-base-lr5e-6-epoch4-final","EffiCoder/deepseek-coder-6.7b-ins-lr5e-6-epoch4-final","EffiCoder/CodeLlama-7b-hf-lr5e-6-epoch4-final","meta-llama/CodeLlama-7b-hf","Qwen/Qwen2.5-7B",,"deepseek-ai/deepseek-coder-6.7b-instruct"
checkpoints = ["meta-llama/Llama-3.1-8B-Instruct"]
def construct_prompt_template(inputs, llm, sampling_params):
    outputs = llm.generate(inputs, sampling_params)
    generated_texts = []
    for i in range(len(outputs)):
        generated_texts.append(outputs[i].outputs[0].text)
    return generated_texts
        
# Function to fetch completion
def fetch_completion(data_entry_lists, llm, sampling_params):
    inputs_batchs = []
    for data_entry in data_entry_lists:
#         inputs_batchs.append(f'''
# "You are an AI programming assistant, utilizing the DeepSeek Coder model, "
# "developed by DeepSeek Company, and you only answer questions related to computer science. "
# "For politically sensitive questions, security and privacy issues, "
# "and other non-computer science questions, you will refuse to answer.\n"

# ```python
# {data_entry['prompt']}
# ```
# ''')
        inputs_batchs.append(f'''
Please continue to complete the function.
{data_entry['prompt']}
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
    dataset = load_dataset("evalplus/humanevalplus",split="test")
    print(checkpoint)
    dataset = [entry for entry in dataset]
    print(dataset[0].keys())
    
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

    if "/sft/" in checkpoint:
        end_name = checkpoint.split("/")[-3]
    else:
        end_name = checkpoint.split("/")[-1]
    with open(f"../../results/humaneval_{end_name}_0.json", "w") as f:
        json.dump(dataset, f, indent=4)


# python vllm_humaneval.py  --checkpoint deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct