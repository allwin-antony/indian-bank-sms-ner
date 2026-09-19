import json
import random
import os

input_file = "data/labeled_sms.jsonl"
train_file = "data/train.jsonl"
val_file = "data/val.jsonl"

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

random.seed(42)
random.shuffle(lines)

split_idx = int(len(lines) * 0.85)
train_lines = lines[:split_idx]
val_lines = lines[split_idx:]

with open(train_file, 'w', encoding='utf-8') as f:
    f.writelines(train_lines)

with open(val_file, 'w', encoding='utf-8') as f:
    f.writelines(val_lines)

print(f"Split {len(lines)} total samples into {len(train_lines)} train and {len(val_lines)} validation.")
