# MSFS Code Lab

**MSFS** —— 一种基于 K-Means 聚类引导的嵌入拉伸（Embedding Stretching）方法，用于缓解文本分类中的**虚假相关性（spurious correlations）**，提升模型在 anti-test（去捷径）测试集上的鲁棒性，同时不牺牲正常测试集的准确率。

**MSFS** — a K-Means-guided **Embedding Stretching** method that mitigates spurious correlations in text classification, improving robustness on anti-test (shortcut-removed) sets without sacrificing standard test accuracy.

中文说明 | [English](#english)

---

## 中文

### 方法概要

MSFS 在微调阶段对"多余样本"的句嵌入做相对固定参考点（**拉伸核心**，stretching core）的按比例拉伸，打破嵌入空间中标签与虚假特征之间的绑定：

1. 用预训练 BERT 对全部训练样本提取嵌入 `z = f_e(x)`；
2. 对嵌入做 **K-Means 聚类**（簇数 `m`，默认 10，配合肘部法则选取）；
3. 取所有簇中心的均值作为**拉伸核心** `s`；
4. **簇内按类向下配平**：在每个簇内统计各类样本数，超过最小类计数 `n_min` 的样本标记为"多余样本"（`IfChangeForKM = 1`）；
5. 微调时，多余样本的池化嵌入被替换为 `z' = s + α(z − s)`（`α` 为拉伸比例；`α < 1` 向核心拉近，`α > 1` 向外推离，`α = 1` 不变），其余样本原样通过；
6. 正常计算损失并更新模型。

评估同时报告 normal-test 与 anti-test 上的 accuracy / precision / recall / F1 以及 **worst-group accuracy**。

### 目录结构

```
MSFS_Code_Lab/
├── MSFT+BERT/                  # MSFS 主方法
│   ├── save_CC.py              # 提取嵌入 → K-Means → 保存拉伸核心 CC.pt，并生成带 IfChangeForKM 标记的训练数据
│   ├── Train.py                # BERT 微调（含嵌入拉伸），输出各项指标
│   ├── Model.py                # KM 模型：在 BertForSequenceClassification 基础上加入拉伸层
│   ├── Function_For_KM.py      # 聚类、配平、拉伸等核心函数
│   ├── extract_collect.py      # 汇总 results/ 下各 α 的结果为 total.csv
│   ├── SH.sh                   # 一键脚本：全部数据集 × 拉伸比例扫描
│   └── bert/                   # BERT-base-uncased 权重与分词器（model.safetensors 走 Git LFS）
├── Ablation_1_Cluster_Num/     # 消融一：聚类数 m 的影响
├── Ablation_2_T-SNE/           # 消融二：嵌入空间 t-SNE 可视化（含纯 BERT 对照组 Train_BERT.py）
├── Ablation_3_Diff_Kernal/     # 消融三：不同拉伸核心的对比（簇均值 vs 全局平均嵌入 vs 随机向量等变体）
├── Ablation_4_Delete_Directly/ # 消融四：直接删除多余样本的对照（MSFSDs）
└── Additional_Data.pdf         # 附录：算法伪代码、拉伸比例分析、Shortcuts Maze 全量结果表
```

### 环境依赖

- Python 3.x，NVIDIA GPU（CUDA）
- `torch`、`transformers`、`datasets`、`sentence-transformers`、`scikit-learn`、`pandas`、`numpy`、`tqdm`、`matplotlib`

> 实验原始环境为 conda 环境 `datawheel`（见各 `.sh` 脚本头部）。

### 快速开始

```bash
# 0) 大文件走 Git LFS，克隆前请先安装并初始化
git lfs install && git clone <本仓库>

# 1) 生成拉伸核心与配平后的训练数据
python save_CC.py \
    --dataset_name /path/to/Concept_Beer_Train.json \
    --model_name ./bert

# 2) 指定拉伸比例训练（α<1 拉近核心，α>1 推离核心）
python Train.py \
    --dataset_name /path/to/Concept_Beer_Train.json \
    --model_name ./bert \
    --strech_proportion 0.1 \
    --test_name /path/to/Concept_Beer_Test.json \
    --cuda_device 0 \
    --num_labels 4

# 3) 汇总结果
python extract_collect.py --dataset_name /path/to/Concept_Beer_Train.json

# 或一键复现全部数据集 × α 扫描
bash SH.sh
```

### 数据与说明

- 数据为 JSONL 格式，每行形如 `{"text": "...", "label": ..., "concepts": ...}`；训练数据不随仓库分发，需自行准备（论文中使用 occurrence / concept / style 三类虚假相关性数据集及其 normal / anti 测试集）。
- `Train.py` 会在运行时为数据自动补充 `IfChangeForKM` 列，并用 `Running.json` 作为实际训练输入。
- 代码中保留了作者原服务器的绝对路径（如 `/home/sql/zlj/...`），运行前请按需修改；`Function_For_KM.py` 中最优簇数 `optimal_k = 10` 为写死值，肘部法则曲线仅作参考。
- 本仓库为科研实验代码（论文附录配套），未做工程化封装，模块间以文件约定耦合。

---

## English

### Overview

MSFS stretches the embeddings of **excess samples** relative to a fixed reference point (the **stretching core**) during fine-tuning, breaking the binding between labels and spurious features in the embedding space:

1. Extract embeddings `z = f_e(x)` of all training samples with a pretrained BERT;
2. Cluster the embeddings with **K-Means** (`m` clusters, default 10, selected with the elbow method);
3. Take the mean of all cluster centers as the **stretching core** `s`;
4. **Per-cluster majority trimming**: within each cluster, samples of a label beyond the minimum class count `n_min` are marked as excess (`IfChangeForKM = 1`);
5. During fine-tuning, the pooled embedding of each excess sample is replaced by `z' = s + α(z − s)` (`α` is the stretching ratio; `α < 1` pulls toward the core, `α > 1` pushes away, `α = 1` is a no-op); all other samples pass through unchanged;
6. Compute the loss and update the model as usual.

Evaluation reports accuracy / precision / recall / F1 and **worst-group accuracy** on both normal-test and anti-test sets.

### Repository Layout

```
MSFS_Code_Lab/
├── MSFT+BERT/                  # Main MSFS method
│   ├── save_CC.py              # Extract embeddings → K-Means → save the stretching core (CC.pt); emit rebalanced data with IfChangeForKM flags
│   ├── Train.py                # BERT fine-tuning with embedding stretching; prints all metrics
│   ├── Model.py                # KM model: BertForSequenceClassification extended with the stretching step
│   ├── Function_For_KM.py      # Core functions: clustering, trimming, stretching
│   ├── extract_collect.py      # Merge per-α results under results/ into total.csv
│   ├── SH.sh                   # One-shot script: all datasets × stretching-ratio sweep
│   └── bert/                   # BERT-base-uncased weights & tokenizer (model.safetensors via Git LFS)
├── Ablation_1_Cluster_Num/     # Ablation 1: effect of the number of clusters m
├── Ablation_2_T-SNE/           # Ablation 2: t-SNE visualization of the embedding space (with a plain-BERT baseline, Train_BERT.py)
├── Ablation_3_Diff_Kernal/     # Ablation 3: alternative stretching cores (cluster-mean vs. global average embedding vs. random vector, etc.)
├── Ablation_4_Delete_Directly/ # Ablation 4: directly deleting excess samples instead of stretching (MSFSDs)
└── Additional_Data.pdf         # Appendix: pseudocode, stretching-ratio analysis, full Shortcuts Maze result tables
```

### Requirements

- Python 3.x, NVIDIA GPU (CUDA)
- `torch`, `transformers`, `datasets`, `sentence-transformers`, `scikit-learn`, `pandas`, `numpy`, `tqdm`, `matplotlib`

> The original experiments ran in a conda environment named `datawheel` (see the header of each `.sh` script).

### Quick Start

```bash
# 0) Large files are stored with Git LFS — install and initialize first
git lfs install && git clone <this-repo>

# 1) Compute the stretching core and the rebalanced training data
python save_CC.py \
    --dataset_name /path/to/Concept_Beer_Train.json \
    --model_name ./bert

# 2) Train with a chosen stretching ratio (α<1 pulls toward the core, α>1 pushes away)
python Train.py \
    --dataset_name /path/to/Concept_Beer_Train.json \
    --model_name ./bert \
    --strech_proportion 0.1 \
    --test_name /path/to/Concept_Beer_Test.json \
    --cuda_device 0 \
    --num_labels 4

# 3) Collect results
python extract_collect.py --dataset_name /path/to/Concept_Beer_Train.json

# Or reproduce the full dataset × α sweep in one shot
bash SH.sh
```

### Data & Notes

- Data files are JSONL, one object per line, e.g. `{"text": "...", "label": ..., "concepts": ...}`. Datasets are **not** distributed with this repository; prepare them separately (the paper uses occurrence / concept / style spurious-correlation datasets with their normal and anti test sets).
- `Train.py` automatically adds the `IfChangeForKM` column at runtime and trains on the generated `Running.json`.
- Absolute paths from the authors' original server (e.g. `/home/sql/zlj/...`) are kept in the code — adjust them before running. The optimal cluster count `optimal_k = 10` in `Function_For_KM.py` is hard-coded; the elbow curve is for reference only.
- This is research code accompanying the paper appendix — minimal engineering packaging, modules are coupled by file conventions.
