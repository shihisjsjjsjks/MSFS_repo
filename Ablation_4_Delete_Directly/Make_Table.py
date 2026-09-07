import os
import pandas as pd
import numpy as np
from pathlib import Path
import re

def process_experiment_results(root_path, output_path):
    """
    处理实验结果并生成表格
    
    Args:
        root_path: 实验结果根目录路径
        output_path: 输出txt文件路径
    """
    
    # 定义所有拉伸比和随机种子
    stretch_ratios = ['0.001', '0.01', '0.1', '10.0', '100.0', '1000.0', '10000.0']
    seeds = ['1', '4', '5']
    
    # 表格列映射关系
    # 根据表格结构：Yelp(Occur:ST,Syn,Catg | Style:Reg,Auth), Emotion(Occur:ST,Syn,Catg | Style:Reg,Auth), Beer(Concept:Occur,Corr)
    dataset_mapping = {
        'Yelp_ST': 'Yelp_Occur_ST',
        'Yelp_Syn': 'Yelp_Occur_Syn', 
        'Yelp_Catg': 'Yelp_Occur_Catg',
        'Yelp_Reg': 'Yelp_Style_Reg',
        'Yelp_Auth': 'Yelp_Style_Auth',
        'Emotion_ST': 'Emotion_Occur_ST',
        'Emotion_Syn': 'Emotion_Occur_Syn',
        'Emotion_Catg': 'Emotion_Occur_Catg',
        'Emotion_Reg': 'Emotion_Style_Reg',
        'Emotion_Auth': 'Emotion_Style_Auth',
        'Beer_Occur': 'Beer_Concept_Occur',
        'Beer_Corr': 'Beer_Concept_Corr'
    }
    
    # 创建存储结果的字典
    test_results = {}  # test测试集结果
    antitest_results = {}  # anti-test测试集结果
    
    # 初始化结果字典
    for ratio in stretch_ratios:
        test_results[ratio] = {}
        antitest_results[ratio] = {}
        for dataset_key in dataset_mapping.keys():
            test_results[ratio][dataset_key] = []
            antitest_results[ratio][dataset_key] = []
    
    # 遍历文件夹
    root_path = Path(root_path)
    if not root_path.exists():
        print(f"错误：路径 {root_path} 不存在")
        return
    
    # 获取所有数据集文件夹
    dataset_folders = [f for f in root_path.iterdir() if f.is_dir()]
    
    for folder in dataset_folders:
        folder_name = folder.name
        
        # 解析数据集名和测试集类型
        # 格式: 数据集名_test 或 数据集名_anti-test
        if folder_name.endswith('_test'):
            test_type = 'test'
            base_dataset_name = folder_name[:-5]  # 去掉'_test'
        elif folder_name.endswith('_anti-test'):
            test_type = 'anti-test'
            base_dataset_name = folder_name[:-10]  # 去掉'_anti-test'
        else:
            print(f"跳过未知格式的文件夹: {folder_name}")
            continue
        
        # 查找对应的表格列
        matched_key = None
        for key in dataset_mapping.keys():
            # 检查base_dataset_name是否以key开头（因为可能有后缀）
            if base_dataset_name.startswith(key):
                matched_key = key
                break
        
        if not matched_key:
            print(f"警告：未找到数据集 {base_dataset_name} 的映射")
            continue
        
        # 遍历所有csv文件
        for csv_file in folder.glob("*.csv"):
            filename = csv_file.stem  # 去掉.csv后缀
            
            # 解析文件名：拉伸比_随机种子
            parts = filename.split('_')
            if len(parts) != 2:
                print(f"跳过格式不正确的文件: {filename}")
                continue
            
            ratio, seed = parts
            
            if ratio not in stretch_ratios or seed not in seeds:
                print(f"跳过未知的拉伸比或随机种子: {filename}")
                continue
            
            # 读取csv文件
            try:
                df = pd.read_csv(csv_file)
                
                # 查找test_f1列
                if 'test_f1' not in df.columns:
                    print(f"文件 {csv_file} 中没有 test_f1 列")
                    continue
                
                # 获取准确率（假设是最后一行的值，或者平均值）
                accuracy_values = df['test_f1'].dropna()
                if len(accuracy_values) == 0:
                    print(f"文件 {csv_file} 中没有有效的准确率数据")
                    continue
                
                # 使用最后一个值（或根据需求调整）
                accuracy = accuracy_values.iloc[-1]
                
                # 存储结果
                if test_type == 'test':
                    test_results[ratio][matched_key].append(accuracy)
                else:  # anti-test
                    antitest_results[ratio][matched_key].append(accuracy)
                    
            except Exception as e:
                print(f"读取文件 {csv_file} 时出错: {e}")
                continue
    
    # 计算统计结果
    test_table_data = {}  # 第二个表格数据
    antitest_table_data = {}  # 第三个表格数据
    
    for ratio in stretch_ratios:
        test_table_data[ratio] = {}
        antitest_table_data[ratio] = {}
        
        for dataset_key in dataset_mapping.keys():
            # test结果
            test_accuracies = test_results[ratio][dataset_key]
            if test_accuracies and len(test_accuracies) > 0:
                mean_val = np.mean(test_accuracies) * 100  # 转换为百分比
                std_val = np.std(test_accuracies) * 100
                test_table_data[ratio][dataset_key] = (mean_val, std_val)
            else:
                test_table_data[ratio][dataset_key] = (np.nan, np.nan)
            
            # anti-test结果
            antitest_accuracies = antitest_results[ratio][dataset_key]
            if antitest_accuracies and len(antitest_accuracies) > 0:
                mean_val = np.mean(antitest_accuracies) * 100
                std_val = np.std(antitest_accuracies) * 100
                antitest_table_data[ratio][dataset_key] = (mean_val, std_val)
            else:
                antitest_table_data[ratio][dataset_key] = (np.nan, np.nan)
    
    # 计算第一个表格的数据（差值的最小值）
    first_table_data = {}
    for dataset_key in dataset_mapping.keys():
        min_diff = float('inf')
        best_ratio = None
        best_test_result = None
        best_antitest_result = None
        
        for ratio in stretch_ratios:
            test_mean, test_std = test_table_data[ratio][dataset_key]
            antitest_mean, antitest_std = antitest_table_data[ratio][dataset_key]
            
            if not (np.isnan(test_mean) or np.isnan(antitest_mean)):
                diff = test_mean - antitest_mean
                if diff < min_diff:
                    min_diff = diff
                    best_ratio = ratio
                    best_test_result = (test_mean, test_std)
                    best_antitest_result = (antitest_mean, antitest_std)
        
        if best_ratio:
            # 格式化结果
            test_formatted = f"{best_test_result[0]:.2f}\\textsubscript{{$\\pm${best_test_result[1]:.2f}}}"
            antitest_formatted = f"{best_antitest_result[0]:.2f}\\textsubscript{{$\\pm${best_antitest_result[1]:.2f}}}"
            first_table_data[dataset_key] = {
                'diff': min_diff,
                'test_result': test_formatted,
                'antitest_result': antitest_formatted
            }
        else:
            first_table_data[dataset_key] = None
    
    # 生成表格文本
    output_lines = []
    
    # 第一个表格
    output_lines.append("% Table 1")
    output_lines.append("\\begin{table*}[htbp]")
    output_lines.append("\\centering")
    output_lines.append("\\caption{Model performance on various datasets}")
    output_lines.append("\\label{tab:model_performance}")
    output_lines.append("\\resizebox{\\textwidth}{!}{%")
    output_lines.append("\\begin{tabular}{@{}l*{12}{c}@{}}")
    output_lines.append("\\toprule")
    output_lines.append("\\textbf{Model} & \\multicolumn{12}{c}{\\textbf{Dataset $\\triangle$ (\\%) Performance}} \\\\")
    output_lines.append("\\cmidrule{2-13}")
    output_lines.append("& \\multicolumn{5}{c}{\\textbf{Yelp}} & \\multicolumn{5}{c}{\\textbf{Emotion}} & \\multicolumn{2}{c}{\\textbf{Beer}} \\\\")
    output_lines.append("\\cmidrule(lr){2-6} \\cmidrule(lr){7-11} \\cmidrule(lr){12-13}")
    output_lines.append("& \\multicolumn{3}{c}{\\textbf{Occur}} & \\multicolumn{2}{c}{\\textbf{Style}} & ")
    output_lines.append("\\multicolumn{3}{c}{\\textbf{Occur}} & \\multicolumn{2}{c}{\\textbf{Style}} & ")
    output_lines.append("\\multicolumn{2}{c}{\\textbf{Concept}} \\\\")
    output_lines.append("\\cmidrule(lr){2-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-9} \\cmidrule(lr){10-11} \\cmidrule(lr){12-13}")
    output_lines.append("& ST & Syn & Catg & Reg & Auth & ST & Syn & Catg & Reg & Auth & Occur & Corr \\\\")
    output_lines.append("\\midrule")
    
    # 填充第一个表格数据
    first_table_row = "MSFT$D_s$ "
    dataset_order = ['Yelp_ST', 'Yelp_Syn', 'Yelp_Catg', 'Yelp_Reg', 'Yelp_Auth',
                    'Emotion_ST', 'Emotion_Syn', 'Emotion_Catg', 'Emotion_Reg', 'Emotion_Auth',
                    'Beer_Occur', 'Beer_Corr']
    
    for dataset_key in dataset_order:
        if first_table_data.get(dataset_key):
            diff = first_table_data[dataset_key]['diff']
            # 格式化差值：保留2位小数，加粗最小值对应的结果
            diff_str = f"\\textbf{{{diff:.2f}}}"
            first_table_row += f"& {diff_str} "
        else:
            first_table_row += "& - "
    
    first_table_row += "\\\\"
    output_lines.append(first_table_row)
    output_lines.append("\\bottomrule")
    output_lines.append("\\end{tabular}%")
    output_lines.append("}")
    output_lines.append("\\end{table*}")
    output_lines.append("")
    
    # 第二个表格（test结果）
    output_lines.append("% Table 2")
    output_lines.append("\\begin{table*}[htbp]")
    output_lines.append("\\centering")
    output_lines.append("\\caption{Method Test Accuracy}")
    output_lines.append("\\label{tab:model_performance}")
    output_lines.append("\\resizebox{\\textwidth}{!}{%")
    output_lines.append("\\begin{tabular}{@{}l*{12}{c}@{}}")
    output_lines.append("\\toprule")
    output_lines.append("\\textbf{Model} & \\multicolumn{12}{c}{\\textbf{Dataset $\\triangle$ (\\%) Performance}} \\\\")
    output_lines.append("\\cmidrule{2-13}")
    output_lines.append("& \\multicolumn{5}{c}{\\textbf{Yelp}} & \\multicolumn{5}{c}{\\textbf{Emotion}} & \\multicolumn{2}{c}{\\textbf{Beer}} \\\\")
    output_lines.append("\\cmidrule(lr){2-6} \\cmidrule(lr){7-11} \\cmidrule(lr){12-13}")
    output_lines.append("& \\multicolumn{3}{c}{\\textbf{Occur}} & \\multicolumn{2}{c}{\\textbf{Style}} & ")
    output_lines.append("\\multicolumn{3}{c}{\\textbf{Occur}} & \\multicolumn{2}{c}{\\textbf{Style}} & ")
    output_lines.append("\\multicolumn{2}{c}{\\textbf{Concept}} \\\\")
    output_lines.append("\\cmidrule(lr){2-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-9} \\cmidrule(lr){10-11} \\cmidrule(lr){12-13}")
    output_lines.append("& ST & Syn & Catg & Reg & Auth & ST & Syn & Catg & Reg & Auth & Occur & Corr \\\\")
    output_lines.append("\\midrule")
    
    # 填充第二个表格数据
    for ratio in stretch_ratios:
        row = f"{ratio} "
        for dataset_key in dataset_order:
            mean_val, std_val = test_table_data[ratio][dataset_key]
            if not np.isnan(mean_val) and not np.isnan(std_val):
                # 检查是否为最小值对应的结果
                if first_table_data.get(dataset_key) and first_table_data[dataset_key].get('test_result'):
                    # 从第一个表格中获取格式化后的字符串
                    formatted = first_table_data[dataset_key]['test_result']
                    # 提取数值部分进行比较
                    test_mean, _ = test_table_data[ratio][dataset_key]
                    if f"{test_mean:.2f}" in formatted:
                        cell_content = f"\\textbf{{{formatted}}}"
                    else:
                        cell_content = f"{mean_val:.2f}\\textsubscript{{$\\pm${std_val:.2f}}}"
                else:
                    cell_content = f"{mean_val:.2f}\\textsubscript{{$\\pm${std_val:.2f}}}"
                row += f"& {cell_content} "
            else:
                row += "& - "
        row += "\\\\"
        output_lines.append(row)
    
    output_lines.append("\\bottomrule")
    output_lines.append("\\end{tabular}%")
    output_lines.append("}")
    output_lines.append("\\end{table*}")
    output_lines.append("")
    
    # 第三个表格（anti-test结果）
    output_lines.append("% Table 3")
    output_lines.append("\\begin{table*}[htbp]")
    output_lines.append("\\centering")
    output_lines.append("\\caption{Method Test Accuracy}")
    output_lines.append("\\label{tab:model_performance}")
    output_lines.append("\\resizebox{\\textwidth}{!}{%")
    output_lines.append("\\begin{tabular}{@{}l*{12}{c}@{}}")
    output_lines.append("\\toprule")
    output_lines.append("\\textbf{Model} & \\multicolumn{12}{c}{\\textbf{Dataset $\\triangle$ (\\%) Performance}} \\\\")
    output_lines.append("\\cmidrule{2-13}")
    output_lines.append("& \\multicolumn{5}{c}{\\textbf{Yelp}} & \\multicolumn{5}{c}{\\textbf{Emotion}} & \\multicolumn{2}{c}{\\textbf{Beer}} \\\\")
    output_lines.append("\\cmidrule(lr){2-6} \\cmidrule(lr){7-11} \\cmidrule(lr){12-13}")
    output_lines.append("& \\multicolumn{3}{c}{\\textbf{Occur}} & \\multicolumn{2}{c}{\\textbf{Style}} & ")
    output_lines.append("\\multicolumn{3}{c}{\\textbf{Occur}} & \\multicolumn{2}{c}{\\textbf{Style}} & ")
    output_lines.append("\\multicolumn{2}{c}{\\textbf{Concept}} \\\\")
    output_lines.append("\\cmidrule(lr){2-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-9} \\cmidrule(lr){10-11} \\cmidrule(lr){12-13}")
    output_lines.append("& ST & Syn & Catg & Reg & Auth & ST & Syn & Catg & Reg & Auth & Occur & Corr \\\\")
    output_lines.append("\\midrule")
    
    # 填充第三个表格数据
    for ratio in stretch_ratios:
        row = f"{ratio} "
        for dataset_key in dataset_order:
            mean_val, std_val = antitest_table_data[ratio][dataset_key]
            if not np.isnan(mean_val) and not np.isnan(std_val):
                # 检查是否为最小值对应的结果
                if first_table_data.get(dataset_key) and first_table_data[dataset_key].get('antitest_result'):
                    # 从第一个表格中获取格式化后的字符串
                    formatted = first_table_data[dataset_key]['antitest_result']
                    # 提取数值部分进行比较
                    antitest_mean, _ = antitest_table_data[ratio][dataset_key]
                    if f"{antitest_mean:.2f}" in formatted:
                        cell_content = f"\\textbf{{{formatted}}}"
                    else:
                        cell_content = f"{mean_val:.2f}\\textsubscript{{$\\pm${std_val:.2f}}}"
                else:
                    cell_content = f"{mean_val:.2f}\\textsubscript{{$\\pm${std_val:.2f}}}"
                row += f"& {cell_content} "
            else:
                row += "& - "
        row += "\\\\"
        output_lines.append(row)
    
    output_lines.append("\\bottomrule")
    output_lines.append("\\end{tabular}%")
    output_lines.append("}")
    output_lines.append("\\end{table*}")
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"表格已生成到: {output_path}")
    
    # 打印一些统计信息
    print(f"\n处理完成:")
    print(f"- 处理的拉伸比: {stretch_ratios}")
    print(f"- 处理的随机种子: {seeds}")
    print(f"- 映射的数据集数量: {len(dataset_mapping)}")

# 使用示例
if __name__ == "__main__":
    # 设置你的文件夹路径和输出文件路径
    input_folder = "/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/results"  # 修改为你的文件夹路径
    output_file = "/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Latex_code.txt"  # 修改为输出文件路径
    
    # 运行处理函数
    process_experiment_results(input_folder, output_file)