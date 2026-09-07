import os
import shutil
import argparse

def move_all_files(source_dir, target_dir, copy_instead=False):
    """
    将源文件夹中的所有文件转移到目标文件夹
    
    参数:
        source_dir: 源文件夹路径
        target_dir: 目标文件夹路径
        copy_instead: 如果为True，则复制而不是移动（保留原文件）
    """
    # 检查源文件夹是否存在
    if not os.path.exists(source_dir):
        print(f"错误: 源文件夹不存在: {source_dir}")
        return False
    
    # 检查源文件夹是否是目录
    if not os.path.isdir(source_dir):
        print(f"错误: 源路径不是文件夹: {source_dir}")
        return False
    
    # 创建目标文件夹（如果不存在）
    os.makedirs(target_dir, exist_ok=True)
    
    # 获取源文件夹中的所有文件（包括子文件夹）
    moved_count = 0
    error_count = 0
    
    # 先处理文件
    for root, dirs, files in os.walk(source_dir):
        # 计算目标路径中的相对路径
        rel_path = os.path.relpath(root, source_dir)
        if rel_path == ".":
            rel_path = ""
        
        # 为每个文件创建对应的目标目录
        for file in files:
            src_file = os.path.join(root, file)
            
            # 构建目标路径
            if rel_path:
                dest_subdir = os.path.join(target_dir, rel_path)
                os.makedirs(dest_subdir, exist_ok=True)
                dest_file = os.path.join(dest_subdir, file)
            else:
                dest_file = os.path.join(target_dir, file)
            
            try:
                if copy_instead:
                    shutil.copy2(src_file, dest_file)  # 复制文件，保留元数据
                else:
                    shutil.move(src_file, dest_file)   # 移动文件
                
                moved_count += 1
                if moved_count % 100 == 0:  # 每100个文件打印一次进度
                    print(f"已处理 {moved_count} 个文件...")
                    
            except Exception as e:
                error_count += 1
                print(f"处理文件失败 {src_file}: {e}")
    
    # 如果不是复制模式，删除源文件夹中的空目录
    if not copy_instead:
        try:
            # 删除源文件夹（如果为空）
            if not os.listdir(source_dir):
                os.rmdir(source_dir)
                print(f"源文件夹已删除（已空）: {source_dir}")
        except Exception as e:
            print(f"删除源文件夹失败: {e}")
    
    print(f"\n转移完成!")
    print(f"成功处理: {moved_count} 个文件")
    print(f"失败: {error_count} 个文件")
    if copy_instead:
        print(f"模式: 复制（源文件夹保留原文件）")
    else:
        print(f"模式: 移动（源文件夹清空）")
    
    return True

def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description="将源文件夹中的所有文件转移到目标文件夹")
    parser.add_argument("source_dir", help="源文件夹路径")
    parser.add_argument("target_dir", help="目标文件夹路径")
    parser.add_argument("--copy", action="store_true", 
                       help="复制模式（保留源文件），默认为移动模式")
    parser.add_argument("--force", action="store_true",
                       help="强制覆盖已存在的目标文件")
    
    args = parser.parse_args()
    
    # 验证路径
    source_dir = os.path.abspath(args.source_dir)
    target_dir = os.path.abspath(args.target_dir)
    
    print(f"源文件夹: {source_dir}")
    print(f"目标文件夹: {target_dir}")
    print(f"模式: {'复制' if args.copy else '移动'}")
    
    # 检查源和目标是否相同
    if source_dir == target_dir:
        print("错误: 源文件夹和目标文件夹不能相同!")
        return
    
    # 执行转移
    success = move_all_files(source_dir, target_dir, copy_instead=args.copy)
    
    if success:
        print("操作完成!")
    else:
        print("操作失败!")

if __name__ == "__main__":
    main()