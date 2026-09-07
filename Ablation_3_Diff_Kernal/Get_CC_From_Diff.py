import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

def compute_average_embedding_from_json(json_path, model_name, save_path, text_key='text'):
    """
    从JSON文件计算所有文本的平均嵌入
    
    参数:
        json_path: JSON文件路径（每行一个JSON对象）
        model_name: SentenceTransformer模型名称
        save_path: 保存平均嵌入的.pt文件路径
        text_key: JSON对象中包含文本的键名
    """
    
    # 1. 加载模型
    print(f"加载模型: {model_name}")
    model = SentenceTransformer(model_name)
    
    # 2. 读取JSON文件
    print(f"读取JSON文件: {json_path}")
    texts = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                try:
                    item = json.loads(line)
                    if text_key in item:
                        texts.append(item[text_key])
                except json.JSONDecodeError as e:
                    print(f"警告: 解析JSON行时出错: {e}")
    
    print(f"找到 {len(texts)} 个文本")
    
    if len(texts) == 0:
        print("错误: 文件中没有找到文本")
        return None
    
    # 3. 计算所有文本的嵌入
    print("计算文本嵌入...")
    all_embeddings = []
    
    # 批量处理以提高效率
    batch_size = 32
    for i in tqdm(range(0, len(texts), batch_size), desc="计算嵌入"):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(
            batch_texts, 
            normalize_embeddings=True,
            show_progress_bar=False
        )
        all_embeddings.append(batch_embeddings)
    
    # 4. 合并所有嵌入
    all_embeddings = np.vstack(all_embeddings)
    
    # 5. 计算平均嵌入
    print("计算平均嵌入...")
    average_embedding = np.mean(all_embeddings, axis=0)
    
    # 转换为PyTorch张量
    average_embedding_tensor = torch.tensor(average_embedding, dtype=torch.float32)
    
    # 6. 保存到.pt文件
    print(f"保存平均嵌入到: {save_path}")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    # 保存张量
    torch.save(average_embedding_tensor, save_path)
    
    # 7. 输出信息
    print(f"\n完成!")
    print(f"处理文本数量: {len(texts)}")
    print(f"嵌入维度: {average_embedding_tensor.shape}")
    print(f"平均嵌入形状: {average_embedding_tensor.shape}")
    
    # 保存一些统计信息
    save_stats = {
        'num_texts': len(texts),
        'embedding_dim': average_embedding_tensor.shape[0],
        'model_name': model_name,
        'json_path': json_path,
        'average_embedding': average_embedding_tensor
    }
    
    stats_path = save_path.replace('.pt', '_stats.pt')
    torch.save(save_stats, stats_path)
    print(f"统计信息已保存到: {stats_path}")
    
    return average_embedding_tensor

# 使用示例
if __name__ == "__main__":
    # 示例1: 基本使用
    json_file = "/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_3_Diff_Kernal/Beer/Corr/train.json"  # 你的JSON文件路径
    model = "google-bert/bert-base-uncased"    # 模型名称
    output_file = "/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_3_Diff_Kernal/average_embedding.pt"  # 输出文件
    
    avg_embedding = compute_average_embedding_from_json(
        json_path=json_file,
        model_name=model,
        save_path=output_file
    )
    
    # 示例2: 加载和验证保存的嵌入
    if avg_embedding is not None:
        print(f"\n加载保存的嵌入进行验证...")
        loaded_embedding = torch.load(output_file)
        print(f"加载的形状: {loaded_embedding.shape}")
        print(f"前10个维度值: {loaded_embedding[:10]}")