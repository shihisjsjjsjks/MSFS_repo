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
parser.add_argument('--tsne_data_file', type=str, required=True, help='JSON file for t-SNE visualization (每行一个JSON对象，包含label, text, IfChangeForKM等字段)')
parser.add_argument('--tsne_output_path', type=str, default='tsne_visualization.png', help='t-SNE visualization output path')

args = parser.parse_args()
dataset_name = args.dataset_name
model_name = args.model_name
strech_proportion = args.strech_proportion
test_name = args.test_name
val_name = args.val_name
cuda_device = args.cuda_device
num_labels = args.num_labels
seed = args.seed
tsne_data_file = args.tsne_data_file
tsne_output_path = args.tsne_output_path
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device

print('='*80)
print(f"dataset_name: {dataset_name}")
print(f"model_name: {model_name}")
print(f"strech_proportion: {strech_proportion}")
print(f"test_name: {test_name}")
print(f"val_name: {val_name}")
print(f"seed: {seed}")
print(f"tsne_data_file: {tsne_data_file}")
print(f"tsne_output_path: {tsne_output_path}")

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
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import colorsys
import glob
import shutil
from sentence_transformers import SentenceTransformer  # 新增导入

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
        self.best_model = None  # 新增：保存最佳模型对象引用
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
                        
                        # 保存模型对象引用（关键修改）
                        self.best_model = trainer.model.to(device)
                        
                        print(f"新的最佳模型已保存: {self.best_model_path}, 验证集准确率: {current_accuracy:.4f}")
                    else:
                        # 如果无法获取trainer，尝试直接保存模型
                        model = kwargs.get('model')
                        if model is not None:
                            model.save_pretrained(self.best_model_path)
                            self.tokenizer.save_pretrained(self.best_model_path)
                            
                            # 保存模型对象引用
                            self.best_model = model.to(device)
                            
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

# =============================================================================
# 新增：t-SNE可视化功能（使用SentenceTransformer方式）
# =============================================================================
print("\n" + "="*80)
print("开始生成t-SNE可视化...")

# 确保输出目录存在
os.makedirs(os.path.dirname(os.path.abspath(tsne_output_path)), exist_ok=True)

def load_jsonl_for_tsne(json_file_path):
    """
    加载用于t-SNE可视化的JSONL文件
    
    Args:
        json_file_path: JSONL文件路径
        
    Returns:
        texts: 文本列表
        labels: 标签列表
        change_flags: IfChangeForKM标记列表
    """
    print(f"正在加载t-SNE数据文件: {json_file_path}")
    
    texts = []
    labels = []
    change_flags = []
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue
                
            try:
                item = json.loads(line)
                
                # 检查必要的字段
                if 'text' not in item:
                    print(f"警告: 第{line_num}行缺少'text'字段，跳过")
                    continue
                
                texts.append(item['text'])
                
                # 获取label（如果有）
                if 'label' in item:
                    labels.append(item['label'])
                else:
                    labels.append(-1)  # 如果没有label，使用-1作为占位符
                
                # 获取IfChangeForKM（如果有）
                if 'IfChangeForKM' in item:
                    change_flags.append(item['IfChangeForKM'])
                else:
                    change_flags.append(0)  # 如果没有，默认为0
                    
            except json.JSONDecodeError as e:
                print(f"错误: 第{line_num}行JSON解析失败: {e}")
                continue
    
    print(f"从 {json_file_path} 加载了 {len(texts)} 个样本")
    print(f"标签分布: 有效标签 {len([l for l in labels if l != -1])}, 缺失标签 {labels.count(-1)}")
    print(f"IfChangeForKM分布: 0={change_flags.count(0)}, 1={change_flags.count(1)}")
    
    return texts, labels, change_flags

def extract_embeddings_sentence_transformer(texts, model_name="google-bert/bert-base-uncased"):
    """
    使用SentenceTransformer提取文本嵌入（与Get_Tensor_List方法完全一致）
    
    Args:
        texts: 文本列表
        model_name: SentenceTransformer模型名称
        
    Returns:
        嵌入向量数组
    """
    print(f"正在加载SentenceTransformer模型: {model_name}")
    
    try:
        # 使用sentence_transformers库加载模型
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"警告: 无法加载模型 {model_name}，错误: {e}")
        print("尝试使用默认模型: google-bert/bert-base-uncased")
        model = SentenceTransformer("google-bert/bert-base-uncased")
    
    # 如果有GPU，使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) == "cuda":
        model = model.to(device)
        print(f"模型已移动到: {device}")
    
    print("正在生成文本嵌入...")
    
    # 使用model.encode方法，与Get_Tensor_List完全一致
    embeddings = model.encode(
        texts, 
        normalize_embeddings=True,  # 与您的normalize_embeddings=True一致
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )
    
    print(f"嵌入生成完成，维度: {embeddings.shape[0]}x{embeddings.shape[1]}")
    print(f"嵌入示例 - 形状: {embeddings[0].shape}, 范数: {np.linalg.norm(embeddings[0]):.6f}")
    
    return embeddings

def extract_embeddings_from_km_model_with_mean_pooling(texts, km_model, tokenizer):
    """
    从KM模型中提取嵌入，使用平均池化（Mean Pooling）而不是CLS池化
    与SentenceTransformer的encode方法保持一致
    
    Args:
        texts: 文本列表
        km_model: KM模型
        tokenizer: 对应的tokenizer
        
    Returns:
        嵌入向量数组
    """
    print("正在从KM模型中提取嵌入（使用平均池化Mean Pooling）...")
    
    # 将模型设置为评估模式
    km_model.eval()
    
    all_embeddings = []
    batch_size = 32
    
    # 逐批次处理
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # 对文本进行编码
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            # 获取所有token的隐藏状态
            bert_outputs = km_model.bert(**inputs)
            last_hidden_state = bert_outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
            
            # 获取attention mask（排除padding tokens）
            attention_mask = inputs['attention_mask']
            
            # 实现平均池化（Mean Pooling）
            # 1. 扩展attention mask的维度以匹配隐藏状态
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            
            # 2. 对非padding token的隐藏状态求和
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            
            # 3. 计算非padding token的数量（确保除数不为0）
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            # 4. 计算平均值
            mean_pooled = sum_embeddings / sum_mask
            
            # 5. L2归一化（与SentenceTransformer保持一致）
            normalized_embeddings = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            
            all_embeddings.append(normalized_embeddings.cpu().numpy())
        
        if (i // batch_size) % 10 == 0:
            print(f"已处理 {min(i+batch_size, len(texts))}/{len(texts)} 个样本")
    
    # 合并所有批次的嵌入
    embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
    
    if len(embeddings) > 0:
        print(f"嵌入提取完成，维度: {embeddings.shape[0]}x{embeddings.shape[1]}")
        print(f"嵌入示例 - 形状: {embeddings[0].shape}, 范数: {np.linalg.norm(embeddings[0]):.6f}")
        print(f"池化方式: 平均池化 (Mean Pooling)")
    else:
        print("警告: 未能提取任何嵌入")
    
    return embeddings

def get_high_contrast_colors(num_colors):
    """
    生成高对比度的颜色列表
    
    Args:
        num_colors: 需要生成的颜色数量
        
    Returns:
        颜色列表，每个颜色为(r, g, b)元组
    """
    colors = []
    
    if num_colors <= 10:
        # 对于少量颜色，使用固定的高对比度颜色
        high_contrast_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        return high_contrast_colors[:num_colors]
    
    elif num_colors <= 20:
        # 使用tab20色彩映射
        cmap = plt.cm.get_cmap('tab20')
        return [cmap(i) for i in range(num_colors)]
    
    else:
        # 对于大量颜色，使用HSL色彩空间均匀分布
        for i in range(num_colors):
            hue = i / num_colors  # 在0-1之间均匀分布
            saturation = 0.7 + 0.2 * (i % 2)  # 交替使用高饱和度和中等饱和度
            lightness = 0.5 + 0.1 * ((i // 2) % 2)  # 交替使用亮度和中等亮度
            
            # 转换为RGB
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            colors.append(rgb)
        
        # 随机打乱颜色顺序以增加视觉区分度
        np.random.seed(42)
        indices = np.random.permutation(len(colors))
        return [colors[i] for i in indices]

def add_color_legend(ax, colors_dict, title="Labels"):
    """
    在图表旁边添加颜色图例
    
    Args:
        ax: matplotlib轴对象
        colors_dict: 标签到颜色的映射字典
        title: 图例标题
    """
    # 创建图例条目
    legend_elements = []
    sorted_labels = sorted(colors_dict.keys())
    
    for label in sorted_labels:
        color = colors_dict[label]
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', 
                      label=str(label),
                      markerfacecolor=color, markersize=10)
        )
    
    # 添加图例
    ax.legend(handles=legend_elements, title=title, 
              bbox_to_anchor=(1.05, 1), loc='upper left',
              fontsize=24, title_fontsize=32)

def create_tsne_visualization(embeddings_2d, labels, change_flags, output_path, seed, model_info="KM Model"):
    """
    创建t-SNE可视化图表
    
    Args:
        embeddings_2d: t-SNE降维后的二维嵌入
        labels: 原始标签列表
        change_flags: IfChangeForKM标志列表
        output_path: 输出图片路径
        seed: 随机种子
        model_info: 模型信息
    """
    print("正在创建t-SNE可视化...")
    
    unique_labels = sorted(set(labels))
    print(f"发现 {len(unique_labels)} 个不同的标签: {unique_labels}")
    
    label_colors_list = get_high_contrast_colors(len(unique_labels))
    label_colors = {label: label_colors_list[i] for i, label in enumerate(unique_labels)}
    
    # 创建包含两个子图的大图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))
    
    # 子图1：根据IfChangeForKM着色
    change_0_indices = [i for i, flag in enumerate(change_flags) if flag == 0]
    change_1_indices = [i for i, flag in enumerate(change_flags) if flag == 1]
    
    # 绘制IfChangeForKM=0的点（绿色）
    if change_0_indices:
        ax1.scatter(
            embeddings_2d[change_0_indices, 0],
            embeddings_2d[change_0_indices, 1],
            c='#2E8B57',
            alpha=0.7,
            s=30,
            label='IfChangeForKM=0',
            edgecolors='#006400',
            linewidths=0.8,
            marker='o'
        )
    
    # 绘制IfChangeForKM=1的点（红色）
    if change_1_indices:
        ax1.scatter(
            embeddings_2d[change_1_indices, 0],
            embeddings_2d[change_1_indices, 1],
            c='#DC143C',
            alpha=0.7,
            s=30,
            label='IfChangeForKM=1',
            edgecolors='#8B0000',
            linewidths=0.8,
            marker='s'
        )
    
    ax1.set_title(f't-SNE by IfChangeForKM (Seed={seed})', fontsize=24, pad=15)
    # 隐藏坐标轴标签和刻度值
    ax1.set_xlabel('')  # 清空x轴标签
    ax1.set_ylabel('')  # 清空y轴标签
    
    # 隐藏刻度值（保留坐标轴线）
    ax1.tick_params(
        axis='both',
        which='both',
        bottom=True,
        top=False,
        left=True,
        right=False,
        labelbottom=False,  # 隐藏底部刻度标签
        labelleft=False     # 隐藏左侧刻度标签
    )
    
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=24, markerscale=1.5)
    
    # 添加统计信息
    change_0_count = len(change_0_indices)
    change_1_count = len(change_1_indices)
    stats_text1 = f'IfChangeForKM=0: {change_0_count} | IfChangeForKM=1: {change_1_count}'
    ax1.text(0.02, 0.98, stats_text1, transform=ax1.transAxes, fontsize=24,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
             facecolor="lightgray", alpha=0.7))
    
    # 子图2：根据原始label着色（使用高对比度颜色）
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
    
    for i, label in enumerate(unique_labels):
        label_indices = [j for j, l in enumerate(labels) if l == label]
        color = label_colors[label]
        marker = markers[i % len(markers)] if len(unique_labels) > 5 else 'o'
        
        if label_indices:
            ax2.scatter(
                embeddings_2d[label_indices, 0],
                embeddings_2d[label_indices, 1],
                c=[color],
                alpha=0.7,
                s=30,
                label=f'{label}',
                edgecolors=color,
                linewidths=0.8,
                marker=marker
            )
    
    ax2.set_title(f't-SNE by Original Label (Seed={seed})', fontsize=24, pad=15)
    # 隐藏坐标轴标签和刻度值
    ax2.set_xlabel('')  # 清空x轴标签
    ax2.set_ylabel('')  # 清空y轴标签
    
    # 隐藏刻度值（保留坐标轴线）
    ax2.tick_params(
        axis='both',
        which='both',
        bottom=True,
        top=False,
        left=True,
        right=False,
        labelbottom=False,  # 隐藏底部刻度标签
        labelleft=False     # 隐藏左侧刻度标签
    )
    
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 添加颜色图例
    add_color_legend(ax2, label_colors, title="Labels")
    
    # 添加大标题
    plt.suptitle(f'KM Model Text Embeddings t-SNE Visualization (Seed={seed}, SP={strech_proportion})', 
                 fontsize=24, y=1.02)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    print(f"正在保存t-SNE可视化到: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"t-SNE可视化已保存到: {output_path}")
    
    return {
        'total_samples': len(labels),
        'change_0_count': change_0_count,
        'change_1_count': change_1_count,
        'unique_labels_count': len(unique_labels)
    }

# 执行t-SNE可视化
try:
    # 加载t-SNE数据
    print(f"从指定文件加载t-SNE数据: {tsne_data_file}")
    tsne_texts, tsne_labels, tsne_change_flags = load_jsonl_for_tsne(tsne_data_file)
    
    print(f"准备为 {len(tsne_texts)} 个样本生成t-SNE可视化")
    
    # 选择要使用的模型
    if best_model_tracker.best_model is not None:
        print("使用训练好的最佳KM模型提取嵌入...")
        best_model_for_embedding = best_model_tracker.best_model
    elif best_model_tracker.best_model_path and os.path.exists(best_model_tracker.best_model_path):
        print(f"从磁盘加载最佳KM模型: {best_model_tracker.best_model_path}")
        best_model_for_embedding = KM.from_pretrained(
            best_model_tracker.best_model_path,
            num_labels=num_labels,
            input_dim=768,
            encoding_dim=128,
            CC=CC,
            strech_proportion=strech_proportion
        ).to(device)
    else:
        print("警告：未找到最佳模型，使用最终训练模型")
        best_model_for_embedding = model
    
    # 从KM模型中提取嵌入（使用平均池化）
    tsne_embeddings = extract_embeddings_from_km_model_with_mean_pooling(
        tsne_texts,
        best_model_for_embedding,
        tokenizer
    )
    
    if len(tsne_embeddings) == 0:
        print("错误: 未能提取嵌入，跳过t-SNE可视化")
    else:
        print(f"嵌入提取完成，形状: {tsne_embeddings.shape}")
        
        # 应用t-SNE降维
        print("正在应用t-SNE降维...")
        
        try:
            tsne = TSNE(
                n_components=2,
                perplexity=min(30, len(tsne_embeddings) - 1),
                random_state=seed,
                max_iter=1000,
                learning_rate='auto',
                init='pca'
            )
            print("使用TSNE新版本参数 (max_iter)")
        except TypeError:
            try:
                tsne = TSNE(
                    n_components=2,
                    perplexity=min(30, len(tsne_embeddings) - 1),
                    random_state=seed,
                    n_iter=1000
                )
                print("使用TSNE旧版本参数 (n_iter)")
            except TypeError as e:
                print(f"警告: TSNE参数错误, 使用默认配置: {e}")
                tsne = TSNE(
                    n_components=2,
                    perplexity=min(30, len(tsne_embeddings) - 1),
                    random_state=seed
                )
        
        embeddings_2d = tsne.fit_transform(tsne_embeddings)
        print(f"t-SNE降维完成, 形状: {embeddings_2d.shape}")
        
        # 创建可视化图表
        model_info = f"KM Model (Val Acc: {best_model_tracker.best_accuracy:.4f}, SP: {strech_proportion})"
        tsne_stats = create_tsne_visualization(
            embeddings_2d,
            tsne_labels,
            tsne_change_flags,
            tsne_output_path,
            seed,
            model_info
        )
        
        print("\n=== t-SNE可视化统计 ===")
        print(f"数据文件: {tsne_data_file}")
        print(f"总样本数: {tsne_stats['total_samples']}")
        print(f"IfChangeForKM=0: {tsne_stats['change_0_count']}")
        print(f"IfChangeForKM=1: {tsne_stats['change_1_count']}")
        print(f"唯一标签数: {tsne_stats['unique_labels_count']}")
        print(f"嵌入维度: {tsne_embeddings.shape[1]}")
        print(f"池化方式: 平均池化 (Mean Pooling)")
        print(f"最佳模型验证准确率: {best_model_tracker.best_accuracy:.4f}")
        print(f"拉伸比例: {strech_proportion}")
        print(f"t-SNE可视化已保存到: {tsne_output_path}")
        
except Exception as e:
    print(f"t-SNE可视化生成失败: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# 清理临时文件以释放空间
# =============================================================================
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
        
        # 清理日志目录
        log_dirs = ["logs", "runs"]
        for log_dir in log_dirs:
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
                print(f"清理日志目录: {log_dir}")
        
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