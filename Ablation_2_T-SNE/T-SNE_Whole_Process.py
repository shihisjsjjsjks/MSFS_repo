import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer  # 改为使用sentence_transformers
from pathlib import Path
import argparse
import warnings
import colorsys

def adjust_tensor_by_proportion(center_tensor, target_tensor, proportion):
    """
    调整目标张量相对于中心张量的位置
    
    Args:
        center_tensor: 中心张量
        target_tensor: 目标张量
        proportion: 调整比例
        
    Returns:
        调整后的目标张量
    """
    # 计算向量差（从中心指向目标的向量）
    difference_vector = target_tensor - center_tensor
    
    # 按给定比例调整目标张量的位置
    adjusted_target_tensor = center_tensor + proportion * difference_vector
    
    return adjusted_target_tensor

def load_jsonl_file(file_path):
    """
    加载JSONL文件（每行一个JSON对象）
    
    Args:
        file_path: JSONL文件路径
        
    Returns:
        包含所有JSON对象的列表
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {e}, 行内容: {line[:100]}")
    return data

def encode_with_sentence_transformer(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    使用SentenceTransformer生成文本嵌入（与Get_Tensor_List方法完全一致）
    
    Args:
        texts: 文本列表
        model_name: 模型名称
        
    Returns:
        嵌入向量数组（维度取决于模型）
    """
    print(f"正在加载SentenceTransformer模型: {model_name}")
    
    try:
        # 使用sentence_transformers库加载模型
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"警告: 无法加载模型 {model_name}，错误: {e}")
        print("尝试使用默认模型: sentence-transformers/all-MiniLM-L6-v2")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
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

def encode_with_bert_sentence_transformer(texts, model_name="google-bert/bert-base-uncased"):
    """
    使用BERT模型通过SentenceTransformer方式生成文本嵌入（768维）
    与Get_Tensor_List方法完全一致
    
    Args:
        texts: 文本列表
        model_name: BERT模型名称
        
    Returns:
        768维嵌入向量数组
    """
    print(f"正在加载BERT SentenceTransformer模型: {model_name}")
    
    try:
        # 注意：对于BERT模型，我们需要使用特定的sentence-transformers版本
        # 或者使用专门为句子嵌入训练的BERT模型
        if "bert-base" in model_name.lower():
            # 尝试加载sentence-transformers版本的BERT
            model = SentenceTransformer("google-bert/bert-base-uncased")
            print("使用google-bert/bert-base-uncased (768维)")
        else:
            # 尝试直接加载
            model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"警告: 无法加载BERT模型 {model_name}，错误: {e}")
        print("使用默认的768维SentenceTransformer模型: google-bert/bert-base-uncased")
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
        batch_size=16,  # BERT模型较大，使用较小的批处理大小
        convert_to_numpy=True
    )
    
    print(f"嵌入生成完成，维度: {embeddings.shape[0]}x{embeddings.shape[1]}")
    print(f"嵌入示例 - 形状: {embeddings[0].shape}, 范数: {np.linalg.norm(embeddings[0]):.6f}")
    
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
              fontsize=9, title_fontsize=10)

def process_and_visualize(json_path, center_tensor_path, SR, output_image_path, 
                         model_name="sentence-transformers/all-MiniLM-L6-v2", use_bert_base=False):
    """
    处理JSON数据并生成t-SNE可视化
    
    Args:
        json_path: JSONL文件路径
        center_tensor_path: 中心张量pt文件路径
        SR: 拉伸比例（0表示不拉伸）
        output_image_path: 输出图片路径
        model_name: 模型名称
        use_bert_base: 是否使用BERT-base（768维）
    """
    
    # 1. 加载数据
    print(f"正在加载数据: {json_path}")
    data = load_jsonl_file(json_path)
    print(f"加载了 {len(data)} 条数据")
    
    # 检查必要字段
    required_fields = ['text', 'label', 'IfChangeForKM']
    for field in required_fields:
        if field not in data[0]:
            raise ValueError(f"数据缺少必要字段: {field}")
    
    # 提取文本和标签
    texts = [item['text'] for item in data]
    labels = [item['label'] for item in data]
    change_flags = [item['IfChangeForKM'] for item in data]
    
    # 2. 生成嵌入（使用SentenceTransformer的model.encode方法）
    if use_bert_base:
        print("使用BERT-base SentenceTransformer模型生成768维嵌入...")
        # 使用专门为句子嵌入训练的BERT模型
        if "google-bert/bert-base-uncased" in model_name:
            # 使用专门为句子相似度训练的BERT模型
            embeddings = encode_with_bert_sentence_transformer(texts, "google-bert/bert-base-uncased")
        else:
            embeddings = encode_with_bert_sentence_transformer(texts, model_name)
    else:
        print("使用SentenceTransformer生成嵌入...")
        embeddings = encode_with_sentence_transformer(texts, model_name)
    
    print(f"嵌入形状: {embeddings.shape}")
    
    # 3. 根据SR值处理嵌入
    if SR != 0:
        print(f"SR={SR}, 正在调整IfChangeForKM=1的嵌入...")
        
        # 加载中心张量
        if not Path(center_tensor_path).exists():
            raise FileNotFoundError(f"中心张量文件不存在: {center_tensor_path}")
        
        center_tensor = torch.load(center_tensor_path)
        center_tensor_np = center_tensor.numpy() if torch.is_tensor(center_tensor) else center_tensor
        
        # 确保中心张量的形状与嵌入匹配
        if center_tensor_np.shape[-1] != embeddings.shape[-1]:
            # 如果维度不匹配，尝试使用第一个嵌入作为中心
            print(f"警告: 中心张量维度({center_tensor_np.shape})与嵌入维度({embeddings.shape})不匹配")
            print(f"使用第一个嵌入作为中心张量")
            center_tensor_np = embeddings[0]
        
        # 调整IfChangeForKM=1的嵌入
        adjusted_embeddings = embeddings.copy()
        
        for i, (embedding, change_flag) in enumerate(zip(embeddings, change_flags)):
            if change_flag == 1:
                # 调整嵌入
                adjusted_embedding = adjust_tensor_by_proportion(
                    center_tensor_np, 
                    embedding, 
                    SR
                )
                adjusted_embeddings[i] = adjusted_embedding
        
        embeddings = adjusted_embeddings
        print("嵌入调整完成")
    else:
        print("SR=0, 不进行嵌入调整")
    
    # 4. 应用t-SNE降维
    print("正在应用t-SNE降维...")
    
    # 根据scikit-learn版本选择合适的参数
    try:
        # 尝试使用新版本的参数名
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, len(embeddings) - 1),  # 确保perplexity不超过数据点数-1
            random_state=42,
            max_iter=1000,  # 新版本使用max_iter
            learning_rate='auto',  # 自动学习率
            init='pca'  # 使用PCA初始化
        )
    except TypeError:
        # 如果失败，尝试使用旧版本的参数名
        try:
            tsne = TSNE(
                n_components=2,
                perplexity=min(30, len(embeddings) - 1),
                random_state=42,
                n_iter=1000  # 旧版本使用n_iter
            )
        except TypeError as e:
            # 如果两个都失败，使用最简单的配置
            print(f"警告: TSNE参数错误, 使用默认配置: {e}")
            tsne = TSNE(
                n_components=2,
                perplexity=min(30, len(embeddings) - 1),
                random_state=42
            )
    
    embeddings_2d = tsne.fit_transform(embeddings)
    print(f"t-SNE降维完成, 形状: {embeddings_2d.shape}")
    
    # 5. 准备标签信息
    unique_labels = sorted(set(labels))
    print(f"发现 {len(unique_labels)} 个不同的标签: {unique_labels}")
    
    # 为每个标签分配高对比度颜色
    label_colors_list = get_high_contrast_colors(len(unique_labels))
    label_colors = {label: label_colors_list[i] for i, label in enumerate(unique_labels)}
    
    # 6. 创建组合可视化
    print("正在创建组合可视化...")
    
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
            c='#2E8B57',  # 海绿色，比纯绿更有区分度
            alpha=0.7,
            s=30,
            label='IfChangeForKM=0',
            edgecolors='#006400',  # 深绿色边框
            linewidths=0.8,
            marker='o'
        )
    
    # 绘制IfChangeForKM=1的点（红色）
    if change_1_indices:
        ax1.scatter(
            embeddings_2d[change_1_indices, 0],
            embeddings_2d[change_1_indices, 1],
            c='#DC143C',  # 深红色，比纯红更有区分度
            alpha=0.7,
            s=30,
            label='IfChangeForKM=1',
            edgecolors='#8B0000',  # 深红色边框
            linewidths=0.8,
            marker='s'  # 使用方形标记以区分
        )
    
    ax1.set_title(f't-SNE by IfChangeForKM (SR={SR})', fontsize=14, pad=15)
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10, markerscale=1.5)
    
    # 添加统计信息
    change_0_count = len(change_0_indices)
    change_1_count = len(change_1_indices)
    stats_text1 = f'IfChangeForKM=0: {change_0_count} | IfChangeForKM=1: {change_1_count}'
    ax1.text(0.02, 0.98, stats_text1, transform=ax1.transAxes, fontsize=10,
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
    
    ax2.set_title(f't-SNE by Original Label (SR={SR})', fontsize=14, pad=15)
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 添加颜色图例
    add_color_legend(ax2, label_colors, title="Labels")
    
    # 添加统计信息
    stats_text2 = f'Total samples: {len(data)}\nUnique labels: {len(unique_labels)}'
    ax2.text(0.02, 0.98, stats_text2, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
             facecolor="lightgray", alpha=0.7))
    
    # 添加大标题
    embedding_dim = embeddings.shape[1]
    model_type = "BERT-base SentenceTransformer" if use_bert_base else "SentenceTransformer"
    plt.suptitle(f'{model_type} Text Embeddings t-SNE Visualization ({embedding_dim}D → 2D, SR={SR})', 
                 fontsize=16, y=1.02)
    
    # 调整布局
    plt.tight_layout()
    
    # 7. 保存图片
    print(f"正在保存图片到: {output_image_path}")
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"处理完成! 图片已保存到: {output_image_path}")
    
    # 返回一些统计信息
    return {
        'total_samples': len(data),
        'change_0_count': change_0_count,
        'change_1_count': change_1_count,
        'unique_labels_count': len(unique_labels),
        'embedding_dim': embeddings.shape[1],
        'tsne_shape': embeddings_2d.shape
    }

def main():
    """
    主函数：解析命令行参数并执行处理
    """
    parser = argparse.ArgumentParser(description='处理文本嵌入并生成t-SNE可视化')
    parser.add_argument('--json_path', type=str, required=True, help='JSONL文件路径')
    parser.add_argument('--center_tensor_path', type=str, required=True, help='中心张量pt文件路径')
    parser.add_argument('--SR', type=float, required=True, help='拉伸比例（0表示不拉伸）')
    parser.add_argument('--output_image_path', type=str, required=True, help='输出图片路径')
    parser.add_argument('--model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2', 
                       help='模型名称（可选，默认：all-MiniLM-L6-v2）')
    parser.add_argument('--use_bert_base', action='store_true',
                       help='使用BERT-base模型（768维）而不是sentence-transformers模型')
    
    args = parser.parse_args()
    
    # 执行处理
    try:
        stats = process_and_visualize(
            args.json_path,
            args.center_tensor_path,
            args.SR,
            args.output_image_path,
            args.model_name,
            args.use_bert_base
        )
        
        # 打印统计信息
        print("\n=== 处理完成 ===")
        print(f"总样本数: {stats['total_samples']}")
        print(f"IfChangeForKM=0: {stats['change_0_count']}")
        print(f"IfChangeForKM=1: {stats['change_1_count']}")
        print(f"唯一标签数: {stats['unique_labels_count']}")
        print(f"嵌入维度: {stats['embedding_dim']}")
        print(f"t-SNE后维度: {stats['tsne_shape']}")
            
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()