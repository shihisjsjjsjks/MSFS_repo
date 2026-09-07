eval "$(conda shell.bash hook)"
conda activate ZLJ
cd /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE

model_name='google-bert/bert-base-uncased'
cuda_device='0'
seeds=(1)

################################## Beer ######################################

train_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/Beer/Corr/train.json"
test_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/Beer/Corr/test.json"
val_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/Beer/Corr/val.json"

num_labels=4

for seed in ${seeds[@]}; do
    python Train_BERT.py \
        --dataset_name "$train_dataset" \
        --model_name "$model_name" \
        --test_name "$test_dataset" \
        --val_name "$val_dataset" \
        --cuda_device "$cuda_device"\
        --num_labels "$num_labels" \
        --seed "$seed"\
        --tsne_data_file /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/KM.json\
        --tsne_output_path "/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/FT_Origin"\
        
done