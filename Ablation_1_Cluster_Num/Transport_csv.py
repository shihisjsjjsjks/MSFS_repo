import os
import shutil
import argparse
import sys

def transfer_csv_files(source_folder, target_folder):
    """
    将源文件夹中的所有csv文件转移到目标文件夹中
    
    Args:
        source_folder: 源文件夹路径（包含csv文件）
        target_folder: 目标文件夹路径（csv文件将被转移到这里）
    """
    # 检查源文件夹是否存在
    if not os.path.exists(source_folder):
        print(f"错误: 源文件夹 '{source_folder}' 不存在")
        return False
    
    # 检查源文件夹是否是文件夹
    if not os.path.isdir(source_folder):
        print(f"错误: '{source_folder}' 不是文件夹")
        return False
    
    # 创建目标文件夹（如果不存在）
    os.makedirs(target_folder, exist_ok=True)
    print(f"目标文件夹 '{target_folder}' 已创建或已存在")
    
    # 获取源文件夹中的所有csv文件
    csv_files = [f for f in os.listdir(source_folder) 
                if f.lower().endswith('.csv') and os.path.isfile(os.path.join(source_folder, f))]
    
    if not csv_files:
        print(f"警告: 源文件夹 '{source_folder}' 中没有csv文件")
        return True
    
    print(f"找到 {len(csv_files)} 个csv文件")
    
    # 转移文件
    transferred_count = 0
    for csv_file in csv_files:
        source_path = os.path.join(source_folder, csv_file)
        target_path = os.path.join(target_folder, csv_file)
        
        try:
            # 如果目标文件已存在，可以选择重命名或跳过
            if os.path.exists(target_path):
                # 方案1: 覆盖文件
                # shutil.copy2(source_path, target_path)
                
                # 方案2: 重命名文件（添加数字后缀）
                base_name = os.path.splitext(csv_file)[0]
                extension = os.path.splitext(csv_file)[1]
                counter = 1
                while os.path.exists(target_path):
                    new_name = f"{base_name}_{counter}{extension}"
                    target_path = os.path.join(target_folder, new_name)
                    counter += 1
                print(f"文件 '{csv_file}' 已存在，重命名为 '{os.path.basename(target_path)}'")
            
            # 转移文件
            shutil.copy2(source_path, target_path)  # copy2会保留文件元数据
            # 如果需要移动而不是复制，可以使用:
            # shutil.move(source_path, target_path)
            
            transferred_count += 1
            print(f"已转移: {csv_file} -> {target_folder}/")
            
        except Exception as e:
            print(f"转移文件 '{csv_file}' 时出错: {e}")
    
    print(f"\n完成! 成功转移 {transferred_count}/{len(csv_files)} 个文件")
    print(f"源文件夹: {source_folder}")
    print(f"目标文件夹: {target_folder}")
    
    return True

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(
        description='将指定源文件夹中的所有csv文件转移到目标文件夹中'
    )
    
    # 方法1: 使用两个单独的参数
    parser.add_argument(
        '--source', '-s', 
        type=str, 
        required=False,  # 设为False，因为我们要支持两种方式
        help='源文件夹路径（包含csv文件）'
    )
    parser.add_argument(
        '--target', '-t', 
        type=str, 
        required=False,  # 设为False，因为我们要支持两种方式
        help='目标文件夹路径（csv文件将被转移到这里）'
    )
    
    # 方法2: 使用单个配置文件或包含两个路径的文件
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='配置文件路径，文件内容应包含两行：第一行源文件夹，第二行目标文件夹'
    )
    
    args = parser.parse_args()
    
    # 确定使用哪种方式获取路径
    if args.config:
        # 从配置文件读取路径
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            if len(lines) < 2:
                print(f"错误: 配置文件 '{args.config}' 需要至少两行（源路径和目标路径）")
                sys.exit(1)
            
            source_folder = lines[0]
            target_folder = lines[1]
            
        except FileNotFoundError:
            print(f"错误: 配置文件 '{args.config}' 不存在")
            sys.exit(1)
        except Exception as e:
            print(f"读取配置文件时出错: {e}")
            sys.exit(1)
            
    elif args.source and args.target:
        # 从命令行参数获取路径
        source_folder = args.source
        target_folder = args.target
        
    else:
        # 如果没有提供足够的参数，显示帮助信息
        parser.print_help()
        print("\n示例用法:")
        print("  1. 使用两个参数:")
        print("     python transfer_csv.py --source /path/to/source --target /path/to/target")
        print("     python transfer_csv.py -s /path/to/source -t /path/to/target")
        print()
        print("  2. 使用配置文件:")
        print("     python transfer_csv.py --config paths.txt")
        print()
        print("  配置文件格式 (paths.txt):")
        print("     /path/to/source/folder")
        print("     /path/to/target/folder")
        sys.exit(1)
    
    # 执行文件转移
    print("=" * 60)
    print("CSV文件转移工具")
    print("=" * 60)
    print(f"源文件夹: {source_folder}")
    print(f"目标文件夹: {target_folder}")
    print("=" * 60)
    
    success = transfer_csv_files(source_folder, target_folder)
    
    if success:
        print("操作完成!")
        sys.exit(0)
    else:
        print("操作失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()