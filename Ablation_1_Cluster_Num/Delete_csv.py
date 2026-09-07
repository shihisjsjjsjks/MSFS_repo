import os
import shutil
import argparse

def clear_folder(folder_path):
    """清空指定文件夹中的所有内容"""
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return False
    
    if not os.path.isdir(folder_path):
        print(f"路径不是文件夹: {folder_path}")
        return False
    
    try:
        # 方法1: 删除并重新创建（更快）
        shutil.rmtree(folder_path)
        os.makedirs(folder_path, exist_ok=True)
        print(f"已清空文件夹: {folder_path}")
        return True
        
    except Exception as e:
        print(f"清空文件夹失败: {e}")
        return False

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description='清空指定文件夹中的所有内容'
    )
    
    parser.add_argument(
        'folder',
        type=str,
        help='要清空的文件夹路径'
    )
    
    # 解析参数
    args = parser.parse_args()
    
    # 执行清空操作
    clear_folder(args.folder)

if __name__ == "__main__":
    main()