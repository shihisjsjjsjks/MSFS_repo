# bin/bash
eval "$(conda shell.bash hook)"
conda activate datawheel

cd /home/sql/zlj/ICLR_MSFT_New/MSFT+BERT

model_name='/home/sql/zlj/ICLR_MSFT_New/MSFT+BERT/bert'
cuda_device='0'
strech_proportions=(0.1 0.5 2 10 100 1000)


train_dataset="/home/sql/zlj/ICLR_MSFT_New/Datasets/Concept_Beer_Train.json"
test_dataset="/home/sql/zlj/ICLR_MSFT_New/Datasets/Concept_Beer_Test.json"


num_labels=4
# 运行保存脚本
python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \


# 循环训练不同参数
for strech_proportion in "${strech_proportions[@]}"; do
    echo "当前参数: strech_proportion=$strech_proportion"
    python Train.py \
        --dataset_name "$train_dataset" \
        --model_name "$model_name" \
        --strech_proportion "$strech_proportion" \
        --test_name "$test_dataset" \
        --cuda_device "$cuda_device"\
        --num_labels "$num_labels"
done

# 提取并收集结果
python extract_collect.py \
    --dataset_name "$train_dataset"