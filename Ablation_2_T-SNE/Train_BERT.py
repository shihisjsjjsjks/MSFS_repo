import os
import argparse

os.environ["WANDB_MODE"] = "disabled"

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset_name', type=str, default='train_52111', help='dataset name')
parser.add_argument('--test_name', type=str, default='Final_Test_Dataset', help='test name')
parser.add_argument('--val_name', type=str, required=True, help='validation dataset name')
parser.add_argument('--model_name', type=str, default='google-bert/bert-base-uncased', help='SentenceTransformer model name')
parser.add_argument('--cuda_device', type=str, default='0', help='cuda device')
parser.add_argument('--num_labels', type=int, default=5, help='total label numbers')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--tsne_output_path', type=str, default='tsne_visualization.png', help='t-SNE visualization output path')
parser.add_argument('--tsne_data_file', type=str, required=True, help='JSON file for t-SNE visualization (每行一个JSON对象，包含label, text, IfChangeForKM等字段)')  # 新增参数

args = parser.parse_args()
seed = args.seed
dataset_name = args.dataset_name
test_name = args.test_name
val_name = args.val_name
model_name = args.model_name
cuda_device = args.cuda_device
num_labels = args.num_labels
tsne_output_path = args.tsne_output_path
tsne_data_file = args.tsne_data_file  # 新增变量
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device

print('='*80)
print(f"seed: {seed}")
print(f"dataset_name: {dataset_name}")
print(f"test_name: {test_name}")
print(f"val_name: {val_name}")
print(f"model_name: {model_name}")
print(f"cuda_device: {cuda_device}")
print(f"tsne_output_path: {tsne_output_path}")
print(f"tsne_data_file: {tsne_data_file}")  # 打印新参数

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from sklearn.metrics import precision_recall_fscore_support
from transformers import TrainerCallback
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch.nn as nn
import random
import json
import shutil
import glob
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import colorsys
from sentence_transformers import SentenceTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    accuracy = np.mean(predictions == labels)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

class MetricsTracker(TrainerCallback):
    def __init__(self, tokenizer, save_dir="model"):
        self.best_val_accuracy = 0.0
        self.best_model_path = None
        self.save_dir = save_dir
        self.tokenizer = tokenizer
        self.best_model = None  # 新增：保存最佳模型对象
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            current_accuracy = metrics['eval_accuracy']
            
            if current_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = current_accuracy
                
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                
                epoch = state.epoch if state.epoch else int(state.global_step / state.max_steps * args.num_train_epochs)
                model_path = os.path.join(self.save_dir, f"best_model_epoch_{epoch:.1f}_acc_{current_accuracy:.4f}")
                
                # 保存模型
                kwargs['model'].save_pretrained(model_path)
                self.tokenizer.save_pretrained(model_path)
                
                # 保存模型对象引用（关键修改）
                self.best_model = kwargs['model'].to(device)
                
                if self.best_model_path and os.path.exists(self.best_model_path):
                    pass
                
                self.best_model_path = model_path
                print(f"新最佳模型已保存: {model_path}, 验证集准确率: {current_accuracy:.4f}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def shuffle_text_in_json(input_path, output_path):
    data = []
    
    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                item = json.loads(line.strip())
                
                if 'text' in item:
                    words = item['text'].split()
                    random.shuffle(words)
                    item['text'] = ' '.join(words)
                
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"解析错误，无法处理行: {line}，错误信息: {e}")
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for item in data:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write('\n')
    
    print(f"文件已保存至: {output_path}")

def get_high_contrast_colors(num_colors):
    colors = []
    
    if num_colors <= 10:
        high_contrast_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        return high_contrast_colors[:num_colors]
    
    elif num_colors <= 20:
        cmap = plt.cm.get_cmap('tab20')
        return [cmap(i) for i in range(num_colors)]
    
    else:
        for i in range(num_colors):
            hue = i / num_colors
            saturation = 0.7 + 0.2 * (i % 2)
            lightness = 0.5 + 0.1 * ((i // 2) % 2)
            
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            colors.append(rgb)
        
        np.random.seed(42)
        indices = np.random.permutation(len(colors))
        return [colors[i] for i in indices]

def add_color_legend(ax, colors_dict, title="Labels"):
    legend_elements = []
    sorted_labels = sorted(colors_dict.keys())
    
    for label in sorted_labels:
        color = colors_dict[label]
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', 
                      label=str(label),
                      markerfacecolor=color, markersize=10)
        )
    
    ax.legend(handles=legend_elements, title=title, 
              bbox_to_anchor=(1.05, 1), loc='upper left',
              fontsize=24, title_fontsize=32)

def extract_embeddings_from_best_model(texts, best_model, tokenizer, batch_size=32):
    """
    从训练好的最佳模型中提取嵌入
    注意：这里需要从分类模型中提取BERT的嵌入表示
    
    Args:
        texts: 文本列表
        best_model: 训练好的最佳分类模型
        tokenizer: 对应的tokenizer
        batch_size: 批处理大小
        
    Returns:
        嵌入向量数组
    """
    print("正在从最佳模型中提取嵌入...")
    
    # 将模型设置为评估模式
    best_model.eval()
    
    all_embeddings = []
    
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
            # 获取模型的BERT输出
            # 注意：这里使用模型的bert部分（对于分类模型，模型内部通常有.bert属性）
            if hasattr(best_model, 'bert'):
                # 对于BERT模型
                outputs = best_model.bert(**inputs)
            elif hasattr(best_model, 'roberta'):
                # 对于RoBERTa模型
                outputs = best_model.roberta(**inputs)
            elif hasattr(best_model, 'distilbert'):
                # 对于DistilBERT模型
                outputs = best_model.distilbert(**inputs)
            else:
                # 如果模型结构不同，尝试获取最后一层隐藏状态
                outputs = best_model(**inputs, output_hidden_states=True)
            
            # 获取最后一层隐藏状态
            if isinstance(outputs, tuple) and len(outputs) > 0:
                if hasattr(outputs[0], 'last_hidden_state'):
                    last_hidden_state = outputs[0].last_hidden_state
                elif isinstance(outputs, dict) and 'last_hidden_state' in outputs:
                    last_hidden_state = outputs['last_hidden_state']
                else:
                    # 尝试从元组中获取
                    last_hidden_state = outputs[0] if isinstance(outputs[0], torch.Tensor) else outputs.last_hidden_state
            else:
                last_hidden_state = outputs.last_hidden_state
            
            # 使用均值池化获取句子嵌入
            # 注意：使用attention mask避免padding token的影响
            attention_mask = inputs['attention_mask']
            
            # 扩展attention mask以匹配隐藏状态的维度
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            
            # 将padding token的贡献设为零
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            
            # 计算均值池化
            mean_embeddings = sum_embeddings / sum_mask
            
            # L2归一化（与SentenceTransformer保持一致）
            normalized_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)
            
            all_embeddings.append(normalized_embeddings.cpu().numpy())
        
        if (i // batch_size) % 10 == 0:
            print(f"已处理 {min(i+batch_size, len(texts))}/{len(texts)} 个样本")
    
    # 合并所有批次的嵌入
    embeddings = np.vstack(all_embeddings)
    
    print(f"嵌入提取完成，维度: {embeddings.shape[0]}x{embeddings.shape[1]}")
    print(f"嵌入示例 - 形状: {embeddings[0].shape}, 范数: {np.linalg.norm(embeddings[0]):.6f}")
    
    return embeddings

def create_tsne_visualization(embeddings_2d, labels, change_flags, output_path, seed, model_info="最佳模型"):
    """
    创建t-SNE可视化图表
    """
    print("正在创建t-SNE可视化...")
    
    unique_labels = sorted(set(labels))
    print(f"发现 {len(unique_labels)} 个不同的标签: {unique_labels}")
    
    label_colors_list = get_high_contrast_colors(len(unique_labels))
    label_colors = {label: label_colors_list[i] for i, label in enumerate(unique_labels)}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))
    
    change_0_indices = [i for i, flag in enumerate(change_flags) if flag == 0]
    change_1_indices = [i for i, flag in enumerate(change_flags) if flag == 1]
    
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
    
    # 隐藏刻度值
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
    
    change_0_count = len(change_0_indices)
    change_1_count = len(change_1_indices)
    stats_text1 = f'IfChangeForKM=0: {change_0_count} | IfChangeForKM=1: {change_1_count}'
    ax1.text(0.02, 0.98, stats_text1, transform=ax1.transAxes, fontsize=24,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
             facecolor="lightgray", alpha=0.7))
    
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
    
    # 隐藏刻度值
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
    
    add_color_legend(ax2, label_colors, title="Labels")
    
    plt.suptitle(f'Fine-tuned Model Text Embeddings t-SNE Visualization (Seed={seed})', 
                 fontsize=24, y=1.02)
    
    plt.tight_layout()
    
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

# 加载数据集
dataset_Train = load_dataset("json", data_files=f"{dataset_name}")
dataset_Val = load_dataset("json", data_files=f"{val_name}")
dataset_Test = load_dataset("json", data_files=f"{test_name}")

# 使用BERT的tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased" if "bert" in model_name.lower() else "distilbert-base-uncased")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else '[PAD]'

def preprocess_function(examples):
    tokenized = tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=512,
        padding="max_length"
    )
    return tokenized

train_dataset = dataset_Train['train'].shuffle(seed=seed)
val_dataset = dataset_Val['train'].shuffle(seed=seed)
test_dataset = dataset_Test['train'].shuffle(seed=seed)

tokenized_dataset_Train = train_dataset.map(preprocess_function, batched=True)
tokenized_dataset_Val = val_dataset.map(preprocess_function, batched=True)
tokenized_dataset_Test = test_dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

classification_model_name = "bert-base-uncased" if "bert" in model_name.lower() else "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(
    classification_model_name, 
    num_labels=num_labels,
    ignore_mismatched_sizes=True
).to(device)

model_save_dir = f"model_{seed}"
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

training_args = TrainingArguments(
    output_dir=model_save_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=False,
    seed=seed,
    report_to='none',
    push_to_hub=False,
    save_total_limit=10,
)

metrics_tracker = MetricsTracker(tokenizer=tokenizer, save_dir=os.path.join(model_save_dir, "best_models"))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_Train,
    eval_dataset=tokenized_dataset_Val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[metrics_tracker]
)

print("开始训练...")
trainer.train()

print("="*80)
print(f"训练完成，最佳验证集准确率: {metrics_tracker.best_val_accuracy:.4f}")

if metrics_tracker.best_model_path:
    print(f"加载最佳模型: {metrics_tracker.best_model_path}")
    
    # 如果已经在MetricsTracker中保存了最佳模型对象，直接使用
    if metrics_tracker.best_model is not None:
        best_model = metrics_tracker.best_model
        print("使用MetricsTracker中保存的最佳模型对象")
    else:
        # 否则从磁盘加载
        best_model = AutoModelForSequenceClassification.from_pretrained(
            metrics_tracker.best_model_path
        ).to(device)
    
    test_trainer = Trainer(
        model=best_model,
        args=TrainingArguments(
            output_dir=model_save_dir,
            per_device_eval_batch_size=32,
            report_to='none',
        ),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("开始测试集评估...")
    test_results = test_trainer.evaluate(tokenized_dataset_Test)
    
    print("测试集最终评价指标：")
    print(f"测试集准确率: {test_results['eval_accuracy']:.4f}")
    print(f"测试集精准率: {test_results['eval_precision']:.4f}")
    print(f"测试集召回率: {test_results['eval_recall']:.4f}")
    print(f"测试集F1分数: {test_results['eval_f1']:.4f}")
    
else:
    print("警告：未找到保存的最佳模型，使用最终模型进行测试")
    print("开始最终测试集评估...")
    test_results = trainer.evaluate(tokenized_dataset_Test)
    
    print("测试集最终评价指标：")
    print(f"测试集准确率: {test_results['eval_accuracy']:.4f}")
    print(f"测试集精准率: {test_results['eval_precision']:.4f}")
    print(f"测试集召回率: {test_results['eval_recall']:.4f}")
    print(f"测试集F1分数: {test_results['eval_f1']:.4f}")

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
test_name_parts = test_name.split('/')
if len(test_name_parts) >= 3:
    test_name_suffix = '_'.join(test_name_parts[-3:])
else:
    test_name_suffix = '_'.join(test_name_parts)
test_name_suffix = test_name_suffix.split('.')[0]

make_dirs(f'SentenceTransformer_results/{test_name_suffix}')

metrics_ls = [
    f'{dataset_name}_{seed}',
    test_results['eval_f1'],
    test_results['eval_accuracy'],
    test_results['eval_precision'],
    test_results['eval_recall'],
]

metrics_df = pd.DataFrame(
    [metrics_ls],
    columns=[
        'dataset_name',
        'test_f1',
        'test_accuracy',
        'test_precision',
        'test_recall',
    ]
)

csv_path = f'SentenceTransformer_results/{test_name_suffix}/{seed}.csv'
metrics_df.to_csv(csv_path, index=False)
print(f"测试结果已保存到: {csv_path}")

if metrics_tracker.best_model_path:
    print(f"最佳模型路径: {metrics_tracker.best_model_path}")
    print(f"最佳验证集准确率: {metrics_tracker.best_val_accuracy:.4f}")

# ========== 修改后的t-SNE可视化逻辑 ==========
print("="*80)
print("开始生成t-SNE可视化...")

os.makedirs(os.path.dirname(os.path.abspath(tsne_output_path)), exist_ok=True)

# 修改：从指定文件加载t-SNE数据
print(f"从指定文件加载t-SNE数据: {tsne_data_file}")
tsne_texts, tsne_labels, tsne_change_flags = load_jsonl_for_tsne(tsne_data_file)

print(f"准备为 {len(tsne_texts)} 个样本生成t-SNE可视化")

# 关键修改：使用训练好的最佳模型提取嵌入
try:
    if metrics_tracker.best_model is not None:
        print("使用训练好的最佳模型提取嵌入...")
        best_model_for_embedding = metrics_tracker.best_model
    elif metrics_tracker.best_model_path:
        print(f"从磁盘加载最佳模型: {metrics_tracker.best_model_path}")
        best_model_for_embedding = AutoModelForSequenceClassification.from_pretrained(
            metrics_tracker.best_model_path
        ).to(device)
    else:
        print("警告：未找到最佳模型，使用最终训练模型")
        best_model_for_embedding = model
    
    # 从最佳模型中提取嵌入（使用指定的数据）
    tsne_embeddings = extract_embeddings_from_best_model(
        tsne_texts,
        best_model_for_embedding,
        tokenizer,
        batch_size=32
    )
    
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
    model_info = f"Fine-tuned Model (Val Acc: {metrics_tracker.best_val_accuracy:.4f})"
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
    print(f"最佳模型验证准确率: {metrics_tracker.best_val_accuracy:.4f}")
    print(f"t-SNE可视化已保存到: {tsne_output_path}")
    
except Exception as e:
    print(f"t-SNE可视化生成失败: {e}")
    import traceback
    traceback.print_exc()

# ========== 清理模型权重文件 ==========
print("="*80)
print("开始清理模型权重文件...")

def clean_model_files(model_dir):
    if os.path.exists(model_dir):
        try:
            shutil.rmtree(model_dir)
            print(f"已删除模型目录: {model_dir}")
            return True
        except Exception as e:
            print(f"删除模型目录 {model_dir} 时出错: {e}")
            return False
    else:
        print(f"模型目录不存在: {model_dir}")
        return False

main_model_dir = model_save_dir
clean_model_files(main_model_dir)

if metrics_tracker.save_dir and os.path.exists(metrics_tracker.save_dir):
    clean_model_files(metrics_tracker.save_dir)

if training_args.output_dir != model_save_dir and os.path.exists(training_args.output_dir):
    clean_model_files(training_args.output_dir)

checkpoint_dirs = [
    f"model_{seed}-checkpoint-*",
    f"{model_save_dir}-checkpoint-*",
]

for pattern in checkpoint_dirs:
    for checkpoint_dir in glob.glob(pattern):
        if os.path.exists(checkpoint_dir):
            try:
                shutil.rmtree(checkpoint_dir)
                print(f"已删除检查点目录: {checkpoint_dir}")
            except Exception as e:
                print(f"删除检查点目录 {checkpoint_dir} 时出错: {e}")

additional_dirs_to_clean = [
    f"runs",
    f"logs",
]

for dir_path in additional_dirs_to_clean:
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
            print(f"已删除临时目录: {dir_path}")
        except Exception as e:
            print(f"删除临时目录 {dir_path} 时出错: {e}")

print("模型权重文件清理完成!")
print("="*80)
print("所有任务完成!")