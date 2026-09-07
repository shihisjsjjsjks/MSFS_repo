import os
import argparse

os.environ["WANDB_MODE"] = "disabled"

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset_name', type=str, default='train_52111', help='dataset name')
parser.add_argument('--model_name', type=str, default='distilbert_base_uncased', help='model name')
parser.add_argument('--strech_proportion', type=float, default=0.1, help='strech proportion')
parser.add_argument('--test_name', type=str, default='Final_Test_Dataset', help='test name')
parser.add_argument('--cuda_device', type=str, default='0', help='cuda device')
parser.add_argument('--num_labels', type=int, default=5, help='total label numbers')

args = parser.parse_args()
dataset_name = args.dataset_name
model_name = args.model_name
strech_proportion = args.strech_proportion
test_name = args.test_name
cuda_device = args.cuda_device
num_labels = args.num_labels
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device

print('='*80)
print(f"dataset_name: {dataset_name}")
print(f"model_name: {model_name}")
print(f"strech_proportion: {strech_proportion}")
print(f"test_name: {test_name}")

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 验证CUDA可用性
if torch.cuda.is_available():
    print(f"Detected {torch.cuda.device_count()} GPU(s):")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA not available! Using CPU")
    
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import TrainerCallback
import logging
from torch import nn
import json
from types import MethodType
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    DistilBertModel,
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    DistilBertConfig
)
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import List, Optional, Tuple, Union
import torch.nn as nn
import torch
import random
import numpy as np
from Model import KM
import pandas as pd

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # 计算整体指标
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = np.mean(predictions == labels)
    
    # 计算每个类别的准确率
    unique_labels = np.unique(labels)
    per_class_acc = []
    for label in unique_labels:
        mask = labels == label
        if np.sum(mask) > 0:  # 确保该类别有样本
            acc = accuracy_score(labels[mask], predictions[mask])
            per_class_acc.append(acc)
    
    # 找到最差组的准确率
    worst_group_acc = min(per_class_acc) if per_class_acc else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'worst_group_acc': worst_group_acc
    }

class MetricsTracker(TrainerCallback):
    def __init__(self):
        self.best_accuracy = 0
        self.total_accuracy = 0
        self.total_precision = 0
        self.total_recall = 0
        self.total_f1 = 0
        self.total_worst_group_acc = 0
        self.num_epochs = 0
        self.count = 0
        self.best_worst_group_acc = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            self.num_epochs += 1
            if 6 <= self.num_epochs <= 10:
                self.count += 1
                self.total_accuracy += metrics['eval_accuracy']
                self.total_precision += metrics['eval_precision']
                self.total_recall += metrics['eval_recall']
                self.total_f1 += metrics['eval_f1']
                self.total_worst_group_acc += metrics['eval_worst_group_acc']
            
            if metrics['eval_accuracy'] > self.best_accuracy:
                self.best_accuracy = metrics['eval_accuracy']
            
            if metrics['eval_worst_group_acc'] > self.best_worst_group_acc:
                self.best_worst_group_acc = metrics['eval_worst_group_acc']

    def get_metrics(self):
        avg_accuracy = self.total_accuracy / self.count if self.count else 0
        avg_precision = self.total_precision / self.count if self.count else 0
        avg_recall = self.total_recall / self.count if self.count else 0
        avg_f1 = self.total_f1 / self.count if self.count else 0
        avg_worst_group_acc = self.total_worst_group_acc / self.count if self.count else 0

        return {
            'best_accuracy': self.best_accuracy,
            'avg_accuracy': avg_accuracy,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1,
            'best_worst_group_acc': self.best_worst_group_acc,
            'avg_worst_group_acc': avg_worst_group_acc
        }

def add_ifchange_column(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        lines = input_file.readlines()

    modified_data = []
    for line in lines:
        try:
            data = json.loads(line.strip())
            data['IfChangeForKM'] = 0
            modified_data.append(data)
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line: {line.strip()}")

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for item in modified_data:
            output_file.write(json.dumps(item, ensure_ascii=False) + '\n')

def shuffle_json_file(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    random.shuffle(lines)
    
    with open(input_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)

shuffle_json_file(f"/home/sql/zlj/ICLR_MSFT_New/MSFT+BERT/Running.json")
add_ifchange_column(f'{test_name}',f"{test_name}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CC = torch.load(f'CC.pt', map_location=device, weights_only=False)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    tokenized = tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=512
    )
    tokenized["if_change"] = examples["IfChangeForKM"]
    return tokenized

class CustomDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        batch["if_change"] = torch.tensor([f["if_change"] for f in features])
        return batch


dataset_train= load_dataset("json", data_files=f"Running.json")['train']
dataset_test= load_dataset("json", data_files=f"{test_name}")['train']

tokenized_train = dataset_train.map(preprocess_function, batched=True)
tokenized_test = dataset_test.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

columns = ["input_ids", "attention_mask", "if_change", "label"]
tokenized_train.set_format(type="torch", columns=columns)
tokenized_test.set_format(type="torch", columns=columns)

batch_size = 64

training_args = TrainingArguments(
    output_dir=f"/home/sql/zlj/ICLR_MSFT_New/MSFT+BERT/{dataset_name}_{model_name}_Model_third_{strech_proportion}",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no",
    load_best_model_at_end=False,
    logging_dir="./logs",
    logging_steps=50,
    report_to="none"
)

model = KM.from_pretrained(
    model_name,
    num_labels=num_labels,
    input_dim=768,
    encoding_dim=128,
    CC=CC,
    strech_proportion=strech_proportion
).to(device)

metrics_tracker = MetricsTracker()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[metrics_tracker]
)

trainer.train()

make_dirs(f'results/{dataset_name}')

metrics = metrics_tracker.get_metrics()

print("最后五轮的评价指标：")
print(f"最高准确率: {metrics['best_accuracy']}")
print(f"平均准确率: {metrics['avg_accuracy']}")
print(f"平均精确率: {metrics['avg_precision']}")
print(f"平均召回率: {metrics['avg_recall']}")
print(f"平均F1分数: {metrics['avg_f1']}")
print(f"最佳最差组准确率: {metrics['best_worst_group_acc']}")
print(f"平均最差组准确率: {metrics['avg_worst_group_acc']}")

metrics_ls = [
    strech_proportion,
    metrics['best_accuracy'],
    metrics['avg_accuracy'],
    metrics['avg_precision'],
    metrics['avg_recall'],
    metrics['avg_f1'],
    metrics['best_worst_group_acc'],
    metrics['avg_worst_group_acc']
]

metrics_df = pd.DataFrame(
    [metrics_ls],
    columns=[
        'strech_proportion',
        'best_accuracy',
        'avg_accuracy',
        'avg_precision',
        'avg_recall',
        'avg_f1',
        'best_worst_group_acc',
        'avg_worst_group_acc'
    ]
)

metrics_df.to_csv(f'results/{dataset_name}/{strech_proportion}.csv', index=False)