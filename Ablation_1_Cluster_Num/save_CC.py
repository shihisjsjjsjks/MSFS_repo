import os
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset_name', type=str, default='Amazon', help='dataset name')
parser.add_argument('--model_name', type=str, default='distilbert_base_uncased', help='model name')
parser.add_argument('--KM_num', type=int)

import torch
import numpy as np
import evaluate
import logging
from torch import nn
from types import MethodType
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split # type: ignore
from transformers import (
    AutoTokenizer,
    BertModel,
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification
)
from Function_For_KM import Rebalancing_and_Stretching
from transformers.modeling_outputs import SequenceClassifierOutput # type: ignore
from typing import List, Optional, Tuple, Union

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"Current GPU: {torch.cuda.current_device()}")
print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

args = parser.parse_args()
model_name = args.model_name
dataset_name = args.dataset_name
KM_num = args.KM_num

print(f"调用 Rebalancing_and_Stretching 函数的参数:")
print(f"model_name: {model_name}")
print(f"dataset_name: {dataset_name}")
print(f"KM_num: {KM_num}")

# 初始化SVM相关数据
Rebalancing_and_Stretching(
    model_name,
    dataset_name,
    KM_num
)