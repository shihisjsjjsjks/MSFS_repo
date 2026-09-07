import os
import argparse

os.environ["WANDB_MODE"] = "disabled"

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset_name', type=str, default='train_52111', help='dataset name')
parser.add_argument('--model_name', type=str, default='distilbert_base_uncased', help='model name')
parser.add_argument('--strech_proportion', type=float, default=0.1, help='strech proportion')
parser.add_argument('--test_name', type=str, default='Final_Test_Dataset', help='test name')
parser.add_argument('--val_name', type=str, required=True, help='validation dataset name')
parser.add_argument('--cuda_device', type=str, default='0', help='cuda device')
parser.add_argument('--num_labels', type=int, default=5, help='total label numbers')
parser.add_argument('--seed', type=int, default=42, help='random seed')

args = parser.parse_args()
dataset_name = args.dataset_name
model_name = args.model_name
strech_proportion = args.strech_proportion
test_name = args.test_name
val_name = args.val_name
cuda_device = args.cuda_device
num_labels = args.num_labels
seed = args.seed
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device

print('='*80)
print(f"dataset_name: {dataset_name}")
print(f"model_name: {model_name}")
print(f"strech_proportion: {strech_proportion}")
print(f"test_name: {test_name}")
print(f"val_name: {val_name}")
print(f"seed: {seed}")

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

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(seed)

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # 计算整体指标 - 使用macro平均
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

class BestModelTracker(TrainerCallback):
    def __init__(self, output_dir, tokenizer):
        self.best_accuracy = 0.0
        self.best_model_path = None
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.best_epoch = 0
        make_dirs(output_dir)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and 'eval_accuracy' in metrics:
            current_accuracy = metrics['eval_accuracy']
            current_epoch = state.epoch
            
            # 如果当前准确率更高，保存为最佳模型
            if current_accuracy > self.best_accuracy:
                self.best_accuracy = current_accuracy
                self.best_epoch = current_epoch
                
                # 删除之前的最佳模型
                if self.best_model_path and os.path.exists(self.best_model_path):
                    import shutil
                    shutil.rmtree(self.best_model_path)
                
                # 保存当前最佳模型
                self.best_model_path = os.path.join(self.output_dir, f"best_model_epoch_{current_epoch}")
                
                try:
                    # 直接保存模型和tokenizer
                    trainer = kwargs.get('trainer')
                    if trainer is not None:
                        # 使用trainer的保存方法
                        trainer.save_model(self.best_model_path)
                        self.tokenizer.save_pretrained(self.best_model_path)
                        print(f"新的最佳模型已保存: {self.best_model_path}, 验证集准确率: {current_accuracy:.4f}")
                    else:
                        # 如果无法获取trainer，尝试直接保存模型
                        model = kwargs.get('model')
                        if model is not None:
                            model.save_pretrained(self.best_model_path)
                            self.tokenizer.save_pretrained(self.best_model_path)
                            print(f"新的最佳模型已保存: {self.best_model_path}, 验证集准确率: {current_accuracy:.4f}")
                        else:
                            print("警告: 无法获取模型对象，跳过模型保存")
                except Exception as e:
                    print(f"保存模型时出错: {e}")
                    # 如果保存失败，清除路径标记
                    self.best_model_path = None

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

# 检查磁盘空间
def check_disk_space(path='.'):
    try:
        stat = os.statvfs(path)
        free_space = stat.f_bavail * stat.f_frsize / (1024 ** 3)  # GB
        print(f"可用磁盘空间: {free_space:.2f} GB")
        return free_space
    except:
        print("无法检查磁盘空间")
        return 10.0  # 假设有足够空间

# 检查磁盘空间
free_space = check_disk_space()
if free_space < 5.0:
    print("警告: 磁盘空间可能不足!")

shuffle_json_file(f"Running.json")
add_ifchange_column(f'{test_name}',f"{test_name}")
add_ifchange_column(f'{val_name}',f"{val_name}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CC = torch.load(f'RT_2.pt', map_location=device, weights_only=False)
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

dataset_train = load_dataset("json", data_files=f"Running.json")['train']
dataset_val = load_dataset("json", data_files=f"{val_name}")['train']
dataset_test = load_dataset("json", data_files=f"{test_name}")['train']

tokenized_train = dataset_train.map(preprocess_function, batched=True)
tokenized_val = dataset_val.map(preprocess_function, batched=True)
tokenized_test = dataset_test.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

columns = ["input_ids", "attention_mask", "if_change", "label"]
tokenized_train.set_format(type="torch", columns=columns)
tokenized_val.set_format(type="torch", columns=columns)
tokenized_test.set_format(type="torch", columns=columns)

batch_size = 32

# 创建输出目录
output_dir = f"Model"
best_model_dir = os.path.join(output_dir, "best_models")
make_dirs(output_dir)
make_dirs(best_model_dir)

# 简化训练参数以减少磁盘使用
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",  
    load_best_model_at_end=False,
    logging_dir="./logs",
    logging_steps=50,
    report_to="none",
    seed=seed,
    save_total_limit=10,
    dataloader_pin_memory=False,  # 减少内存使用
)

model = KM.from_pretrained(
    model_name,
    num_labels=num_labels,
    input_dim=768,
    encoding_dim=128,
    CC=CC,
    strech_proportion=strech_proportion
).to(device)

# 创建最佳模型跟踪器
best_model_tracker = BestModelTracker(best_model_dir, tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,  # 确保tokenizer被传入
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[best_model_tracker]
)

# 训练模型
try:
    trainer.train()
except Exception as e:
    print(f"训练过程中出错: {e}")
    print("尝试继续执行测试...")

print("训练完成，开始加载最佳模型进行测试集评估...")

# 检查最佳模型
if best_model_tracker.best_model_path and os.path.exists(best_model_tracker.best_model_path):
    print(f"加载最佳模型: {best_model_tracker.best_model_path} (epoch {best_model_tracker.best_epoch})")
    try:
        best_model = KM.from_pretrained(
            best_model_tracker.best_model_path,
            num_labels=num_labels,
            input_dim=768,
            encoding_dim=128,
            CC=CC,
            strech_proportion=strech_proportion
        ).to(device)
        
        # 使用最佳模型创建新的Trainer进行测试
        best_trainer = Trainer(
            model=best_model,
            args=TrainingArguments(
                output_dir=output_dir,
                per_device_eval_batch_size=batch_size,
                report_to="none"
            ),
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )
        
        # 在测试集上进行评估
        test_results = best_trainer.evaluate(tokenized_test)
        
    except Exception as e:
        print(f"加载最佳模型失败: {e}")
        print("使用最终模型进行测试")
        test_results = trainer.evaluate(tokenized_test)
else:
    print("未找到最佳模型，使用最终模型进行测试")
    test_results = trainer.evaluate(tokenized_test)

print("测试集最终评价指标：")
print(f"测试集准确率: {test_results['eval_accuracy']:.4f}")
print(f"测试集精确率: {test_results['eval_precision']:.4f}")
print(f"测试集召回率: {test_results['eval_recall']:.4f}")
print(f"测试集F1分数: {test_results['eval_f1']:.4f}")
if hasattr(best_model_tracker, 'best_accuracy'):
    print(f"最佳模型验证集准确率: {best_model_tracker.best_accuracy:.4f} (epoch {best_model_tracker.best_epoch})")

# 提取test_name路径的后三段并拼接
test_name_parts = test_name.split('/')
if len(test_name_parts) >= 3:
    test_name_suffix = '_'.join(test_name_parts[-3:])
else:
    test_name_suffix = '_'.join(test_name_parts)
test_name_suffix = test_name_suffix.split('.')[0]

make_dirs(f'results/{test_name_suffix}')

# 记录测试集的最终指标
metrics_ls = [
    strech_proportion,
    seed,
    test_results['eval_accuracy'],
    test_results['eval_precision'],
    test_results['eval_recall'],
    test_results['eval_f1'],
    best_model_tracker.best_accuracy if hasattr(best_model_tracker, 'best_accuracy') else 0.0
]

metrics_df = pd.DataFrame(
    [metrics_ls],
    columns=[
        'strech_proportion',
        'seed',
        'test_accuracy',
        'test_precision',
        'test_recall',
        'test_f1',
        'best_val_accuracy'
    ]
)

metrics_df.to_csv(f'results/{test_name_suffix}/{strech_proportion}_{seed}.csv', index=False)

print(f"测试结果已保存到: results/{test_name_suffix}/{strech_proportion}_{seed}.csv")

# 清理临时文件以释放空间
try:
    import shutil
    # 删除训练过程中产生的临时checkpoint
    if os.path.exists(output_dir):
        for item in os.listdir(output_dir):
            if item.startswith("checkpoint-"):
                checkpoint_path = os.path.join(output_dir, item)
                shutil.rmtree(checkpoint_path)
                print(f"已清理临时checkpoint: {checkpoint_path}")
except Exception as e:
    print(f"清理临时文件时出错: {e}")

# =============================================================================
# 清理所有模型文件（在代码运行所有完成后）
# =============================================================================
print("\n" + "="*80)
print("开始清理所有模型文件...")

def cleanup_model_files():
    """清理本次运行产生的所有模型文件"""
    try:
        import shutil
        
        # 要清理的目录列表
        directories_to_clean = [
            output_dir,  # 主输出目录
            best_model_dir,  # 最佳模型目录
        ]
        
        cleaned_count = 0
        total_size = 0
        
        for directory in directories_to_clean:
            if os.path.exists(directory):
                print(f"清理目录: {directory}")
                
                # 计算目录大小
                dir_size = 0
                for dirpath, dirnames, filenames in os.walk(directory):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        try:
                            dir_size += os.path.getsize(filepath)
                        except:
                            pass
                
                # 删除目录
                shutil.rmtree(directory)
                cleaned_count += 1
                total_size += dir_size
                print(f"  已删除，释放空间: {dir_size / (1024**3):.2f} GB")
        
        if cleaned_count > 0:
            print(f"\n清理完成！共删除 {cleaned_count} 个目录，释放空间: {total_size / (1024**3):.2f} GB")
        else:
            print("没有需要清理的模型文件")
            
    except Exception as e:
        print(f"清理模型文件时出错: {e}")

# 执行清理
cleanup_model_files()

print("="*80)
print("所有任务完成！")
print("="*80)