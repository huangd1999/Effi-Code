import json
import os
import glob
import numpy as np

def calculate_memory_usage(dat_file_path):
    with open(dat_file_path, 'r') as file:
        prev_time = 0
        prev_mem_mb = 0
        mem_time_mb_s = 0
        next(file)
        for line in file:
            if "__main__." in line:
                continue
            parts = line.split()
            mem_in_mb = float(parts[1])
            timestamp = float(parts[2])
            if prev_time > 0:
                time_interval_s = timestamp - prev_time
                mem_time_mb_s += (prev_mem_mb + mem_in_mb) / 2 * time_interval_s
            prev_time = timestamp
            prev_mem_mb = mem_in_mb
        return mem_time_mb_s


def calculate_runtime(dat_file_path):
    with open(dat_file_path, 'r') as file:
        start_time = float("inf")
        end_time = float("-inf")
        next(file)
        for line in file:
            if "__main__." in line:
                continue
            parts = line.split()
            timestamp = float(parts[2])
            start_time = min(start_time, timestamp)
            end_time = max(end_time, timestamp)
        return max(end_time - start_time,0)

def report_max_memory_usage(dat_file_path):
    max_memory_usage = 0
    with open(dat_file_path, 'r') as file:
        next(file)
        for line in file:
            if "__main__." in line:
                continue
            parts = line.split()
            mem_in_mb = float(parts[1])
            max_memory_usage = max(max_memory_usage, mem_in_mb)
        return max_memory_usage
model_list = [
            "canonical_solution",
            "deepseek-coder-6.7b-base",
            "deepseek-coder-6.7b-instruct",
            "Qwen/Qwen2.5-Coder-7B",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "final_sft_deepseek-coder-6.7b-base",
            "final_sft_deepseek-coder-6.7b-instruct",
            "final_sft_Qwen2.5-Coder-7B",
            "final_sft_Qwen2.5-Coder-7B-Instruct",
    ]
canonical_solution_directory = "../../overheads/humaneval_canonical_solution_0"
canonical_solution_memory_usage = {}
canonical_solution_execution_time = {}
canonical_solution_max_memory_usage = {}
for dat_file in glob.glob(os.path.join(canonical_solution_directory, "*.dat")):
    try:
        problem_idx = os.path.basename(dat_file).split('.')[0]
        canonical_solution_memory_usage[int(problem_idx)] = calculate_memory_usage(dat_file)
        canonical_solution_execution_time[int(problem_idx)] = calculate_runtime(dat_file)
        canonical_solution_max_memory_usage[int(problem_idx)] = report_max_memory_usage(dat_file)
    except:
        pass


global_result = {}
step = 5
pass_list = {}
for model in model_list:
    if "/sft/" in model:
        model = model.split("/")[-3]
    else:
        model = model.split("/")[-1]
    completion_memory_usage = {}
    execution_time = {}
    max_memory_usage = {}
    task_idx = {}
    pass_model = 0
    dat_directory = f"../overheads/humaneval_{model}_0"
    for dat_file in glob.glob(os.path.join(dat_directory, "*.dat")):
        pass_model+=1
        if "final_sft_" in dat_file:
            if not os.path.exists(dat_file.replace("final_sft_","")) :
                continue
        else:
            if not os.path.exists(dat_file.replace(f"../overheads/humaneval_{model}_0",f"../overheads/humaneval_final_sft_{model}_0")):
                continue
        try:
            tmp_model = model
            if "_" in model:
                tmp_model = model.split("_")[0]
            problem_idx = os.path.basename(dat_file).split('.')[0]
            execution_time_result = calculate_runtime(dat_file)
            completion_memory_usage[int(problem_idx)] = calculate_memory_usage(dat_file)
            execution_time[int(problem_idx)] = calculate_runtime(dat_file)
            max_memory_usage[int(problem_idx)] = report_max_memory_usage(dat_file)
            task_idx[int(problem_idx)] = dat_file
        except Exception as e:

            print(dat_file)
    pass_list[model] = pass_model/164*100
    global_result[model] = {"completion_memory_usage":completion_memory_usage,"execution_time":execution_time,"max_memory_usage":max_memory_usage,"task_idx":task_idx}

global_subset = {}

for model in global_result.keys():
    completion_memory_usage = global_result[model]["completion_memory_usage"]
    task_idx = global_result[model]["task_idx"]
    for idx in completion_memory_usage.keys():
        if idx not in global_subset.keys():
            global_subset[idx] = 1
        else:
            global_subset[idx] += 1

for model in global_result.keys():
    completion_memory_usage = global_result[model]["completion_memory_usage"]
    execution_time = global_result[model]["execution_time"]
    max_memory_usage = global_result[model]["max_memory_usage"]

    # report execution time
    total_execution_time = 0

    # report normalized execution time
    normalized_execution_time = 0

    # report max memory usage
    total_max_memory_usage = 0

    # report normalized max memory usage
    normalized_max_memory_usage = 0

    # report memory usage
    total_memory_usage = 0
    total_canonical_solution_max_memory_usage = 0
    total_canonical_solution_execution_time = 0
    total_canonical_solution_memory_usage = 0
    # report normalized memory usage
    normalized_memory_usage = 0
    total_codes = 0
    normalized_execution_time_list = []
    normalized_max_memory_usage_list = []
    normalized_memory_usage_list = []

    helper = 0
    for idx in completion_memory_usage.keys():
        if idx not in canonical_solution_memory_usage.keys():
            continue
        helper+=1
        total_memory_usage += completion_memory_usage[idx]
        total_execution_time += execution_time[idx]
        total_max_memory_usage += max_memory_usage[idx]
        total_canonical_solution_max_memory_usage+=canonical_solution_max_memory_usage[idx]
        total_canonical_solution_memory_usage+=canonical_solution_memory_usage[idx]
        total_canonical_solution_execution_time+=canonical_solution_execution_time[idx]
        normalized_execution_time += execution_time[idx]/canonical_solution_execution_time[idx]
        normalized_execution_time_list.append(execution_time[idx]/canonical_solution_execution_time[idx])
        normalized_max_memory_usage += max_memory_usage[idx]/canonical_solution_max_memory_usage[idx]
        normalized_max_memory_usage_list.append(max_memory_usage[idx]/canonical_solution_max_memory_usage[idx])
        net = execution_time[idx] / canonical_solution_execution_time[idx]
        nmu = completion_memory_usage[idx] / canonical_solution_memory_usage[idx]
        ntmu = max_memory_usage[idx] / canonical_solution_max_memory_usage[idx]
        normalized_memory_usage += completion_memory_usage[idx]/canonical_solution_memory_usage[idx]
        normalized_memory_usage_list.append(completion_memory_usage[idx]/canonical_solution_memory_usage[idx])


    if len(normalized_execution_time_list)==0:
        normalized_execution_time_list.append(1)
        helper = 1
    normalized_execution_time = normalized_execution_time/len(normalized_execution_time_list)
    normalized_max_memory_usage = normalized_max_memory_usage/len(normalized_execution_time_list)
    normalized_memory_usage = normalized_memory_usage/len(normalized_execution_time_list)
    total_execution_time = total_execution_time/helper
    total_memory_usage = total_memory_usage/helper
    total_max_memory_usage = total_max_memory_usage/helper
    pass1 = len(completion_memory_usage)/164*100
    if "final_sft_" not in model:
        print(f"{model}&{total_execution_time:.2f}&{normalized_execution_time:.2f}&{total_max_memory_usage:.2f}&{normalized_max_memory_usage:.2f}&{total_memory_usage:.2f}&{normalized_memory_usage:.2f}&{pass1:.1f}&{pass_list[model]:.1f}\\\\")
    else:
        print(f"+ SFT&{total_execution_time:.2f}&{normalized_execution_time:.2f}&{total_max_memory_usage:.2f}&{normalized_max_memory_usage:.2f}&{total_memory_usage:.2f}&{normalized_memory_usage:.2f}&{pass1:.1f}&{pass_list[model]:.1f}\\\\")