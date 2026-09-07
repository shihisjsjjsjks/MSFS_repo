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

python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \


for seed in ${seeds[@]}; do
    python Train.py \
        --dataset_name "$train_dataset" \
        --test_name "$test_dataset" \
        --val_name "$val_dataset" \
        --tsne_data_file /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/KM.json \
        --tsne_output_path FT_0.001.png \
        --strech_proportion 0.001 \
        --model_name "$model_name" \
        --seed 1
        
done


for seed in ${seeds[@]}; do
    python Train.py \
        --dataset_name "$train_dataset" \
        --test_name "$test_dataset" \
        --val_name "$val_dataset" \
        --tsne_data_file /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/KM.json \
        --tsne_output_path FT_0.01.png \
        --strech_proportion 0.01 \
        --model_name "$model_name" \
        --seed 1
        
done


for seed in ${seeds[@]}; do
    python Train.py \
        --dataset_name "$train_dataset" \
        --test_name "$test_dataset" \
        --val_name "$val_dataset" \
        --tsne_data_file /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/KM.json \
        --tsne_output_path FT_0.1.png \
        --strech_proportion 0.1 \
        --model_name "$model_name" \
        --seed 1
        
done


for seed in ${seeds[@]}; do
    python Train.py \
        --dataset_name "$train_dataset" \
        --test_name "$test_dataset" \
        --val_name "$val_dataset" \
        --tsne_data_file /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/KM.json \
        --tsne_output_path FT_10.png \
        --strech_proportion 10 \
        --model_name "$model_name" \
        --seed 1
        
done

for seed in ${seeds[@]}; do
    python Train.py \
        --dataset_name "$train_dataset" \
        --test_name "$test_dataset" \
        --val_name "$val_dataset" \
        --tsne_data_file /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/KM.json \
        --tsne_output_path FT_100.png \
        --strech_proportion 100 \
        --model_name "$model_name" \
        --seed 1
        
done

for seed in ${seeds[@]}; do
    python Train.py \
        --dataset_name "$train_dataset" \
        --test_name "$test_dataset" \
        --val_name "$val_dataset" \
        --tsne_data_file /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/KM.json \
        --tsne_output_path FT_1000.png \
        --strech_proportion 1000 \
        --model_name "$model_name" \
        --seed 1
        
done

for seed in ${seeds[@]}; do
    python Train.py \
        --dataset_name "$train_dataset" \
        --test_name "$test_dataset" \
        --val_name "$val_dataset" \
        --tsne_data_file /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/KM.json \
        --tsne_output_path FT_10000.png \
        --strech_proportion 10000 \
        --model_name "$model_name" \
        --seed 1
        
done