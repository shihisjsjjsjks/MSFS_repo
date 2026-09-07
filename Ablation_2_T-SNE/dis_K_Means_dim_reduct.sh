eval "$(conda shell.bash hook)"
conda activate ZLJ

#google-bert/bert-base-uncased

cd /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE

model_name='google-bert/bert-base-uncased'
train_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/Beer/Corr/train.json"

python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \


python /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/T-SNE_Whole_Process.py \
  --json_path /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/Running.json \
  --center_tensor_path /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/CC.pt \
  --SR 0 \
  --output_image_path Before_Fine_Tuned/T-SNE_Origin.png \
  --use_bert_base \
  --model_name google-bert/bert-base-uncased


python /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/T-SNE_Whole_Process.py \
  --json_path /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/Running.json \
  --center_tensor_path /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/CC.pt \
  --SR 0.001 \
  --output_image_path Before_Fine_Tuned/T-SNE_0.001.png \
  --use_bert_base \
  --model_name google-bert/bert-base-uncased



python /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/T-SNE_Whole_Process.py \
  --json_path /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/Running.json \
  --center_tensor_path /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/CC.pt \
  --SR 0.01 \
  --output_image_path Before_Fine_Tuned/T-SNE_0.01.png \
  --use_bert_base \
  --model_name google-bert/bert-base-uncased



python /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/T-SNE_Whole_Process.py \
  --json_path /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/Running.json \
  --center_tensor_path /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/CC.pt \
  --SR 0.1 \
  --output_image_path Before_Fine_Tuned/T-SNE_0.1.png \
  --use_bert_base \
  --model_name google-bert/bert-base-uncased




python /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/T-SNE_Whole_Process.py \
  --json_path /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/Running.json \
  --center_tensor_path /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/CC.pt \
  --SR 10 \
  --output_image_path Before_Fine_Tuned/T-SNE_10.png \
  --use_bert_base \
  --model_name google-bert/bert-base-uncased




python /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/T-SNE_Whole_Process.py \
  --json_path /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/Running.json \
  --center_tensor_path /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/CC.pt \
  --SR 100 \
  --output_image_path Before_Fine_Tuned/T-SNE_100.png \
  --use_bert_base \
  --model_name google-bert/bert-base-uncased



python /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/T-SNE_Whole_Process.py \
  --json_path /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/Running.json \
  --center_tensor_path /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/CC.pt \
  --SR 1000 \
  --output_image_path Before_Fine_Tuned/T-SNE_1000.png \
  --use_bert_base \
  --model_name google-bert/bert-base-uncased



python /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/T-SNE_Whole_Process.py \
  --json_path /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/Running.json \
  --center_tensor_path /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/CC.pt \
  --SR 10000 \
  --output_image_path Before_Fine_Tuned/T-SNE_10000.png \
  --use_bert_base \
  --model_name google-bert/bert-base-uncased