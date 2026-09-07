eval "$(conda shell.bash hook)"
conda activate ZLJ

cd /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_3_Diff_Kernal

model_name='google-bert/bert-base-uncased'
cuda_device='0'
strech_proportions=(0.001 0.01 0.1 10 100 1000 10000)
seeds=(1 4 5)


train_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_3_Diff_Kernal/Beer/Corr/train.json"
test_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_3_Diff_Kernal/Beer/Corr/test.json"
val_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_3_Diff_Kernal/Beer/Corr/val.json"


num_labels=4

python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train_random_embedding.py \
            --dataset_name "$train_dataset" \
            --model_name "$model_name" \
            --strech_proportion "$strech_proportion" \
            --test_name "$test_dataset" \
            --val_name "$val_dataset"\
            --cuda_device "$cuda_device"\
            --num_labels "$num_labels"\
            --seed "$seed"
    done
done


python Teleport_folder.py /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_3_Diff_Kernal/results/Beer_Corr_test /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_3_Diff_Kernal/Beer_Corr_test_random_embedding









python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train_average_embedding.py \
            --dataset_name "$train_dataset" \
            --model_name "$model_name" \
            --strech_proportion "$strech_proportion" \
            --test_name "$test_dataset" \
            --val_name "$val_dataset"\
            --cuda_device "$cuda_device"\
            --num_labels "$num_labels"\
            --seed "$seed"
    done
done



python Teleport_folder.py /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_3_Diff_Kernal/results/Beer_Corr_test /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_3_Diff_Kernal/Beer_Corr_test_average_embedding













python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train_RT_1.py \
            --dataset_name "$train_dataset" \
            --model_name "$model_name" \
            --strech_proportion "$strech_proportion" \
            --test_name "$test_dataset" \
            --val_name "$val_dataset"\
            --cuda_device "$cuda_device"\
            --num_labels "$num_labels"\
            --seed "$seed"
    done
done



python Teleport_folder.py /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_3_Diff_Kernal/results/Beer_Corr_test /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_3_Diff_Kernal/Beer_Corr_test_RT_1
















python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train_RT_2.py \
            --dataset_name "$train_dataset" \
            --model_name "$model_name" \
            --strech_proportion "$strech_proportion" \
            --test_name "$test_dataset" \
            --val_name "$val_dataset"\
            --cuda_device "$cuda_device"\
            --num_labels "$num_labels"\
            --seed "$seed"
    done
done





python Teleport_folder.py /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_3_Diff_Kernal/results/Beer_Corr_test /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_3_Diff_Kernal/Beer_Corr_test_RT_2













python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train_RT_3.py \
            --dataset_name "$train_dataset" \
            --model_name "$model_name" \
            --strech_proportion "$strech_proportion" \
            --test_name "$test_dataset" \
            --val_name "$val_dataset"\
            --cuda_device "$cuda_device"\
            --num_labels "$num_labels"\
            --seed "$seed"
    done
done


python Teleport_folder.py /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_3_Diff_Kernal/results/Beer_Corr_test /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_3_Diff_Kernal/Beer_Corr_test_RT_3




















































train_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_3_Diff_Kernal/Beer/Corr/train.json"
test_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_3_Diff_Kernal/Beer/Corr/anti-test.json"
val_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_3_Diff_Kernal/Beer/Corr/val.json"


num_labels=4

python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train_random_embedding.py \
            --dataset_name "$train_dataset" \
            --model_name "$model_name" \
            --strech_proportion "$strech_proportion" \
            --test_name "$test_dataset" \
            --val_name "$val_dataset"\
            --cuda_device "$cuda_device"\
            --num_labels "$num_labels"\
            --seed "$seed"
    done
done


python Teleport_folder.py /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_3_Diff_Kernal/results/Beer_Corr_anti-test /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_3_Diff_Kernal/Beer_Corr_anti-test_random_embedding









python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train_average_embedding.py \
            --dataset_name "$train_dataset" \
            --model_name "$model_name" \
            --strech_proportion "$strech_proportion" \
            --test_name "$test_dataset" \
            --val_name "$val_dataset"\
            --cuda_device "$cuda_device"\
            --num_labels "$num_labels"\
            --seed "$seed"
    done
done



python Teleport_folder.py /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_3_Diff_Kernal/results/Beer_Corr_anti-test /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_3_Diff_Kernal/Beer_Corr_anti-test_average_embedding













python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train_RT_1.py \
            --dataset_name "$train_dataset" \
            --model_name "$model_name" \
            --strech_proportion "$strech_proportion" \
            --test_name "$test_dataset" \
            --val_name "$val_dataset"\
            --cuda_device "$cuda_device"\
            --num_labels "$num_labels"\
            --seed "$seed"
    done
done



python Teleport_folder.py /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_3_Diff_Kernal/results/Beer_Corr_anti-test /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_3_Diff_Kernal/Beer_Corr_anti-test_RT_1
















python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train_RT_2.py \
            --dataset_name "$train_dataset" \
            --model_name "$model_name" \
            --strech_proportion "$strech_proportion" \
            --test_name "$test_dataset" \
            --val_name "$val_dataset"\
            --cuda_device "$cuda_device"\
            --num_labels "$num_labels"\
            --seed "$seed"
    done
done





python Teleport_folder.py /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_3_Diff_Kernal/results/Beer_Corr_anti-test /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_3_Diff_Kernal/Beer_Corr_anti-test_RT_2













python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train_RT_3.py \
            --dataset_name "$train_dataset" \
            --model_name "$model_name" \
            --strech_proportion "$strech_proportion" \
            --test_name "$test_dataset" \
            --val_name "$val_dataset"\
            --cuda_device "$cuda_device"\
            --num_labels "$num_labels"\
            --seed "$seed"
    done
done


python Teleport_folder.py /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_3_Diff_Kernal/results/Beer_Corr_anti-test /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_3_Diff_Kernal/Beer_Corr_anti-test_RT_3

python Make_F.py