import json
from transformers import AutoTokenizer
import numpy as np

# 1. 토크나이저 경로를 현재 모델에 맞게 수정
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# 2. mt_bench 폴더 안에 있는 결과 파일 경로로 수정
jsonl_file = "mt_bench/llama31_8b_instruct_ea-temperature-0.0.jsonl"
jsonl_file_base = "mt_bench/llama31_8b_instruct-temperature-0.0.jsonl"

data = []
with open(jsonl_file, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)

speeds = []
for datapoint in data:
    tokens = sum(datapoint["choices"][0]['new_tokens'])
    times = sum(datapoint["choices"][0]['wall_time'])
    if times > 0:
        speeds.append(tokens / times)

data = []
with open(jsonl_file_base, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)

speeds0 = []
for datapoint in data:
    answer = datapoint["choices"][0]['turns']
    tokens = 0
    for i in answer:
        # Llama-3 토크나이저로 생성된 토큰 수 계산
        tokens += (len(tokenizer.encode(i, add_special_tokens=False)))
    times = sum(datapoint["choices"][0]['wall_time'])
    if times > 0:
        speeds0.append(tokens / times)

print(f"EAGLE 속도: {np.array(speeds).mean():.2f} tokens/s")
print(f"Base 속도: {np.array(speeds0).mean():.2f} tokens/s")
print(f"가속 비율: {np.array(speeds).mean() / np.array(speeds0).mean():.2f}배")