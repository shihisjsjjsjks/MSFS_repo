eval "$(conda shell.bash hook)"
conda activate ZLJ

cd /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_1_Cluster_Num

model_name='google-bert/bert-base-uncased'
cuda_device='0'
strech_proportions=(0.001 0.01 0.1 10 100 1000 10000)
KM_num_array=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)  # 改成数组形式
seeds=(1)

train_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_1_Cluster_Num/Beer/Corr/train.json"
test_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_1_Cluster_Num/Beer/Corr/test.json"
val_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_1_Cluster_Num/Beer/Corr/val.json"

num_labels=4

# 遍历 KM_num
for KM_num in "${KM_num_array[@]}"; do
    echo "===================================================================="
    echo "开始处理 KM_num = $KM_num"
    echo "===================================================================="
    
    # 执行 save_CC.py
    python save_CC.py \
        --dataset_name "$train_dataset" \
        --model_name "$model_name" \
        --KM_num "$KM_num"
    
    echo "save_CC.py 执行完成，开始执行 Train.py"
    
    # 遍历种子和拉伸比例
    for seed in "${seeds[@]}"; do
        for strech_proportion in "${strech_proportions[@]}"; do
            echo "当前参数: KM_num=$KM_num, seed=$seed, strech_proportion=$strech_proportion"
            python Train.py \
                --dataset_name "$train_dataset" \
                --model_name "$model_name" \
                --strech_proportion "$strech_proportion" \
                --test_name "$test_dataset" \
                --val_name "$val_dataset" \
                --cuda_device "$cuda_device" \
                --num_labels "$num_labels" \
                --seed "$seed"
        done
    done

    python Transport_csv.py \
        --source /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_1_Cluster_Num/results/Beer_Corr_test\
        --target /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_1_Cluster_Num/Summary_Results/test/KM_${KM_num}


    python Delete_csv.py /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_1_Cluster_Num/results/Beer_Corr_test\

    
    echo "KM_num = $KM_num 处理完成"
    echo ""
done

echo "所有 KM_num 处理完成！"





















train_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_1_Cluster_Num/Beer/Corr/train.json"
test_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_1_Cluster_Num/Beer/Corr/anti-test.json"
val_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_1_Cluster_Num/Beer/Corr/val.json"

num_labels=4

# 遍历 KM_num
for KM_num in "${KM_num_array[@]}"; do
    echo "===================================================================="
    echo "开始处理 KM_num = $KM_num"
    echo "===================================================================="
    
    # 执行 save_CC.py
    python save_CC.py \
        --dataset_name "$train_dataset" \
        --model_name "$model_name" \
        --KM_num "$KM_num"
    
    echo "save_CC.py 执行完成，开始执行 Train.py"
    
    # 遍历种子和拉伸比例
    for seed in "${seeds[@]}"; do
        for strech_proportion in "${strech_proportions[@]}"; do
            echo "当前参数: KM_num=$KM_num, seed=$seed, strech_proportion=$strech_proportion"
            python Train.py \
                --dataset_name "$train_dataset" \
                --model_name "$model_name" \
                --strech_proportion "$strech_proportion" \
                --test_name "$test_dataset" \
                --val_name "$val_dataset" \
                --cuda_device "$cuda_device" \
                --num_labels "$num_labels" \
                --seed "$seed"
        done
    done

    python Transport_csv.py \
        --source /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_1_Cluster_Num/results/Beer_Corr_anti-test\
        --target /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_1_Cluster_Num/Summary_Results/anti-test/KM_${KM_num}


    python Delete_csv.py /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_1_Cluster_Num/results/Beer_Corr_anti-test\

    
    echo "KM_num = $KM_num 处理完成"
    echo ""
done

echo "所有 KM_num 处理完成！"







python Make_F.py