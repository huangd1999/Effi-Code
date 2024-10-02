# Effi-Code: Unleashing Code Efficiency in Language Models

## Installation

```bash
cd Effi-Code
pip install -r requirements.txt
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

## Fine-tune LLMs with Effi-Code

We have provided the Effi-Code dataset in Effi-Code/LLaMA-Factory/data/efficoder.json, so we can directly use SFT and other methods to finetune LLMs.

```bash
cd Effi-Code/LLaMA-Factory
bash run.sh
```

## VLLM inference 

```bash
cd Effi-Code/scripts
bash run.sh
```

## Report Efficiency and pass@1 results

```bash
cd Effi-Code/src
python code_efficiency_calculator.py
python calculate_memory_usage.py
```