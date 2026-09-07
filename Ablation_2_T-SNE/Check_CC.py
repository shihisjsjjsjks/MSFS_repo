import torch

def get_tensor_shape(pt_file_path):
    """
    获取pt文件中张量的形状
    
    Args:
        pt_file_path: pt文件路径
    
    Returns:
        张量形状的元组，如果不是张量则返回None
    """
    try:
        # 加载文件
        data = torch.load(pt_file_path)
        
        # 检查是否是张量
        if torch.is_tensor(data):
            print(f"张量形状: {data.shape}")
            print(f"张量维度: {len(data.shape)}D")
            return data.shape
        else:
            print(f"文件内容不是PyTorch张量，类型: {type(data)}")
            return None
            
    except Exception as e:
        print(f"加载文件失败: {e}")
        return None

# 使用示例
if __name__ == "__main__":
    # 替换为你的pt文件路径
    pt_file = "/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/CC.pt"
    shape = get_tensor_shape(pt_file)
    
    if shape:
        print(f"形状为: {shape}")