import json
import os

def remove_lines_with_ifchange(json_file_path, output_file_path=None, key_to_check='IfChangeForKM', target_value=1):
    """
    删除JSON文件中指定键值为特定值的行
    
    Args:
        json_file_path: 输入JSON文件路径
        output_file_path: 输出文件路径（如果为None，则覆盖原文件）
        key_to_check: 要检查的键名，默认为'IfChangeForKM'
        target_value: 要删除的目标值，默认为1
    """
    # 如果没有指定输出文件路径，则创建一个临时文件名，最后覆盖原文件
    if output_file_path is None:
        output_file_path = json_file_path + '.tmp'
    
    removed_count = 0
    kept_count = 0
    
    try:
        # 打开输入文件和输出文件
        with open(json_file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                    
                try:
                    # 解析JSON对象
                    json_obj = json.loads(line)
                    
                    # 检查是否包含指定的键且值为目标值
                    if key_to_check in json_obj and json_obj[key_to_check] == target_value:
                        removed_count += 1
                        # 不写入这一行（即删除）
                    else:
                        # 写入保留的行
                        outfile.write(line + '\n')
                        kept_count += 1
                        
                except json.JSONDecodeError as e:
                    print(f"警告: 第 {line_num} 行不是有效的JSON格式: {e}")
                    # 如果是无效JSON行，可以选择保留或跳过
                    # 这里选择保留原行
                    outfile.write(line + '\n')
                    kept_count += 1
        
        # 如果输出文件路径是临时文件，覆盖原文件
        if output_file_path != json_file_path:
            os.replace(output_file_path, json_file_path)
            
        print(f"处理完成:")
        print(f"  删除了 {removed_count} 行（{key_to_check} == {target_value})")
        print(f"  保留了 {kept_count} 行")
        print(f"  原文件已更新")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {json_file_path}")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        # 清理临时文件
        if os.path.exists(output_file_path) and output_file_path != json_file_path:
            os.remove(output_file_path)

# 使用示例
if __name__ == "__main__":
    # 示例1: 直接修改原文件
    input_file = "Running.json"
    
    # 调用函数，不指定输出文件路径则会覆盖原文件
    remove_lines_with_ifchange(input_file)
    
    # 示例2: 保存到新文件
    # remove_lines_with_ifchange("input.json", "output.json")
    
    # 示例3: 检查其他键值
    # remove_lines_with_ifchange("input.json", "output.json", key_to_check='some_key', target_value='delete_me')