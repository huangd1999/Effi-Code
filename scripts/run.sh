python ../src/vllm_humaneval.py --checkpoint deepseek-ai/deepseek-coder-6.7b-base
python ../src/vllm_humaneval.py --checkpoint deepseek-ai/deepseek-coder-6.7b-instruct
python ../src/vllm_humaneval.py --checkpoint Qwen/Qwen2.5-Coder-7B
python ../src/vllm_humaneval.py --checkpoint Qwen/Qwen2.5-Coder-7B-Instruct
python ../src/vllm_humaneval.py --checkpoint ../LLaMA-Factory/saves/final_sft_deepseek-coder-6.7b-instruct
python ../src/vllm_humaneval.py --checkpoint ../LLaMA-Factory/saves/final_sft_deepseek-coder-6.7b-base
python ../src/vllm_humaneval.py --checkpoint ../LLaMA-Factory/saves/final_sft_Qwen2.5-Coder-7B
python ../src/vllm_humaneval.py --checkpoint ../LLaMA-Factory/saves/final_sft_Qwen2.5-Coder-7B-Insruct
python ../src/vllm_effibench.py --checkpoint deepseek-ai/deepseek-coder-6.7b-base
python ../src/vllm_effibench.py --checkpoint deepseek-ai/deepseek-coder-6.7b-instruct
python ../src/vllm_effibench.py --checkpoint Qwen/Qwen2.5-Coder-7B
python ../src/vllm_effibench.py --checkpoint Qwen/Qwen2.5-Coder-7B-Instruct
python ../src/vllm_effibench.py --checkpoint ../LLaMA-Factory/saves/final_sft_deepseek-coder-6.7b-instruct
python ../src/vllm_effibench.py --checkpoint ../LLaMA-Factory/saves/final_sft_deepseek-coder-6.7b-base
python ../src/vllm_effibench.py --checkpoint ../LLaMA-Factory/saves/final_sft_Qwen2.5-Coder-7B
python ../src/vllm_effibench.py --checkpoint ../LLaMA-Factory/saves/final_sft_Qwen2.5-Coder-7B-Insruct

