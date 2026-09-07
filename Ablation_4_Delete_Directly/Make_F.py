import os

def create_empty_file_in_current_dir(filename='F.txt'):
    """
    在当前脚本所在目录创建空白文件
    """
    # 获取当前脚本所在的目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建完整的文件路径
    file_path = os.path.join(current_dir, filename)
    
    try:
        # 创建空白文件
        with open(file_path, 'w', encoding='utf-8') as f:
            pass  # 不写入任何内容，创建空文件
        
        print(f"已成功在 '{current_dir}' 创建空白文件 '{filename}'")
        return True
        
    except Exception as e:
        print(f"创建文件失败: {e}")
        return False

# 调用函数
if __name__ == "__main__":
    create_empty_file_in_current_dir()