import os
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from scipy.spatial.distance import cdist
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
#实现目标：输入模型以及输入目标数据集路径，通过一个函数运算可以得到经过算法处理后的数据集（按概念分类向下配平+多余部分张量拉近）
import torch.nn as nn
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_center_tensor(tensor_list):
    """
    计算张量列表在几何空间中的中心张量。
    
    参数：
    tensor_list (list of torch.Tensor): 张量列表，每个张量的形状相同
    
    返回：
    torch.Tensor: 中心张量
    """
    # 确保所有张量的形状一致
    tensor_shapes = [t.shape for t in tensor_list]
    if len(set(tensor_shapes)) != 1:
        raise ValueError("所有张量必须具有相同的形状")

    # 将所有张量加起来并除以张量的数量
    sum_tensor = torch.zeros_like(torch.from_numpy(tensor_list[0]), dtype=torch.float32)
    for t in tensor_list:
        sum_tensor += t
    
    # 计算中心张量
    center_tensor = sum_tensor / len(tensor_list)
    
    return center_tensor

def kmeans_clustering_with_elbow(tensor_list, KM_num, save_path="elbow_curve.png"):
    """
    使用肘部法则进行K-Means聚类，并返回聚类结果以及每个聚类的中心张量。
    
    参数:
    tensor_list (list of np.ndarray): 张量列表，每个张量形状为[1, 768]
    save_path (str): 保存肘部法则图像的路径，默认是当前目录下的 "elbow_curve.png"
    
    返回:
    kmeans.labels_: 聚类标签
    kmeans.cluster_centers_: 聚类中心（每个聚类的中心张量）
    kmeans: 聚类模型
    """
    
    # 将所有张量转换为二维数组，每个张量的形状为(1, 768)，因此需要堆叠成(总样本数, 768)
    X = np.vstack(tensor_list)
    
    # # 计算不同聚类数下的SSE (误差平方和)
    sse = []
    k_range = range(1, 20)  # 聚类数从1到10
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)
    
    # 绘制肘部法则曲线
    plt.figure(figsize=(8, 6))
    plt.plot(k_range, sse, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.grid(True)
    
    # 将图像保存到本地路径
    plt.savefig(save_path)
    print(f"肘部法则图像已保存到: {save_path}")
    plt.close()  # 关闭图像，以免影响后续显示

    # 选择最佳的聚类数 (通过终端选择肘部法则曲线中的“肘部”位置)
    optimal_k = KM_num
    
    # 使用最佳聚类数进行K-Means聚类
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(X)
    # print(kmeans.labels_)
    # 获取每个聚类的中心张量
    cluster_centers = kmeans.cluster_centers_
    
    CC = compute_center_tensor(cluster_centers)
    CC = CC.cpu().numpy()
    CC = torch.from_numpy(CC).to(device=device, dtype=torch.float32)

    # 新增保存代码
    torch.save(CC.cpu(), 'CC.pt')  # 保存到CPU张量格式
    
    return kmeans.labels_, cluster_centers, kmeans

def predict_cluster(new_tensor, cluster_centers):
    """
    根据新输入的张量预测它属于哪个聚类。
    
    参数:
    new_tensor (np.ndarray): 形状为[1, 768]的新张量
    cluster_centers (np.ndarray): 聚类中心的张量数组
    
    返回:
    int: 新张量所属的聚类标签
    """
    
    # 计算新张量与每个聚类中心的距离
    distances = cdist(new_tensor, cluster_centers, metric='euclidean')
    
    # 返回距离最近的聚类标签
    return np.argmin(distances)

def add_labels_to_comments(comments, kmeans_labels):
    """
    将标签列表一一对应地添加到评论列表中，并新建相应的字段输出JSON格式。
    
    参数:
    comments (list of str): 评论列表，每个元素是一个评论字符串
    labels (list of int): 标签列表，每个元素是对应评论的标签
    concepts (list of str): 概念列表，每个元素是对应评论的概念（可选）
    if_change (int): 如果有改变，设为1，否则为0
    
    返回:
    list: 每个评论的字典形式，符合所需的JSON格式
    """
    
    result = []
    
    # 为每个评论创建一个字典
    for i in range(len(comments)):
        result.append({
            "label": comments[i]['label'],
            "text": comments[i]['text'],
            "concepts": comments[i]['concepts'],
            # "IfChange": comments.if_change[i],
            "Cluster": kmeans_labels[i]  # 假设Cluster与标签是一样的，可以根据实际调整
        })
    
    return result

def Get_Tensor_List(reviews_list,Tensor_list,model):
    for item in tqdm(reviews_list, total=reviews_list.__len__()):
        text = item['text']
        q_embeddings = model.encode(text, normalize_embeddings=True)
        Tensor_list.append(q_embeddings)

def Get_Review_list(reviews_list,Datasets_Path):
    with open(Datasets_Path, 'r', encoding='utf-8') as f:
        for line in f:
            review = json.loads(line.strip())
            reviews_list.append(review)

def Rebalancing_and_Stretching(model_name,Datasets_Path,KM_num):
    reviews_list = []#获取评论列表
    Get_Review_list(reviews_list,Datasets_Path)

    model = SentenceTransformer(model_name)

    Tensor_list=[]#获取张量列表
    Get_Tensor_List(reviews_list,Tensor_list,model)

    searching_for_conceptual_redundancy(reviews_list,Tensor_list,KM_num)

def Update_Raw_Datasets(original_data, record_data, index_column='index', ifchange_column='IfChange'):
    # 将原始数据集和记载数据集转化为DataFrame
    original_df = pd.DataFrame(original_data)
    record_df = pd.DataFrame(record_data)
    
    # 按照index_column的顺序重新排列记载数据集
    record_df_sorted = record_df.sort_values(by=[index_column]).reset_index(drop=True)
    
    # 检查原始数据集是否包含IfChange列，如果没有就创建
    if ifchange_column not in original_df.columns:
        original_df[ifchange_column] = None  # 或者可以初始化为其他默认值
    
    # 将记载数据集中的IfChange列数据写入原始数据集中的IfChange列
    original_df[ifchange_column] = record_df_sorted[ifchange_column].values
    save_dataframe_to_json(original_df,r'Running.json')

    return original_df

def searching_for_conceptual_redundancy(review, tensors, KM_num):
    kmeans_labels, cluster_centers, kmeans = kmeans_clustering_with_elbow(tensors, KM_num) 
    
    my_label_setlist = list(range(len(cluster_centers)))
    label_set = set(my_label_setlist)  
    
    labels = {row['label'] for row in review}
    dataset_label_num = len(labels)
    
    list_A = {label: {'Cluster': [], 'label': [],'index':[]} for label in label_set} 
    list_A_Iftomin = {i1: {i2: [] for i2 in labels} for i1 in label_set} 
    
    id = 0
    for data, label in zip(review, kmeans_labels):
        list_A[label]['index'].append(id)
        list_A[label]['label'].append(data['label'])
        list_A[label]['Cluster'].append(label)
        id += 1
    
    balanced_list = []
    excess_list = []
    
    for i in range(len(label_set)):
        label = list_A[i]['label']
        Index = list_A[i]['index']
        
        label_count = defaultdict(int)
        
        for lbl in label:
            label_count[lbl] += 1

        min_count = min(label_count.values())  
        
        balanced_data = []
        excess_data = []
        
        for m in range(dataset_label_num):
            list_A_Iftomin[i][m] = 0
        
        count_Indexs = 0
        for lbl,id in tqdm(zip(label, Index), total=len(list_A[i]), desc=f"for label {i}"):
            if  list_A_Iftomin[i][lbl] == min_count:
                IfChangeForKM = 1
                excess_data.append({'label': lbl,'index': Index[count_Indexs], 'IfChangeForKM':IfChangeForKM})
                count_Indexs += 1
            else:
                IfChangeForKM = 0
                balanced_data.append({'label': lbl,'index': Index[count_Indexs], 'IfChangeForKM':IfChangeForKM})
                list_A_Iftomin[i][lbl] += 1
                count_Indexs += 1

        balanced_list.extend(balanced_data)
        excess_list.extend(excess_data)
    
    balanced_list.extend(excess_list) 
    Update_Raw_Datasets(review, balanced_list,'index', 'IfChangeForKM')

def adjust_tensor_by_proportion(center_tensor, target_tensor, proportion):
    # 计算向量差（从中心指向目标的向量）
    difference_vector = target_tensor - center_tensor
    
    # 按给定比例调整目标张量的位置
    adjusted_target_tensor = center_tensor + proportion * difference_vector
    
    return adjusted_target_tensor

def save_dataframe_to_json(df, filename):
    df.to_json(filename, orient='records', lines=True)
    print("数据已逐行保存到指定文件中")

def push_away(tensor, target_tensor, weight):
    """
    将当前张量在空间上远离目标张量。
    
    参数：
    tensor (torch.Tensor): 当前张量
    target_tensor (torch.Tensor): 目标张量
    weight (float): 拉远的距离比重,应该大于1
    
    返回：
    torch.Tensor: 拉远后的张量
    """
    # 计算张量之间的差异
    diff = tensor - target_tensor
    
    # 根据比重来调整张量的差异，推远当前张量
    new_tensor = tensor + weight * diff
    
    return new_tensor
