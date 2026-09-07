eval "$(conda shell.bash hook)"
conda activate ZLJ

cd /root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly

model_name='google-bert/bert-base-uncased'
cuda_device='0'
strech_proportions=(0.001 0.01 0.1 10 100 1000 10000)
seeds=(1)


train_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Beer/Corr/train.json"
test_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Beer/Corr/test.json"
val_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Beer/Corr/val.json"


num_labels=4

python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \

python Delete_Excess.py


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train.py \
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








train_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Beer/Corr/train.json"
test_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Beer/Corr/anti-test.json"
val_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Beer/Corr/val.json"


num_labels=4

python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \

python Delete_Excess.py


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train.py \
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














train_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Beer/Occur/train.json"
test_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Beer/Occur/test.json"
val_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Beer/Occur/val.json"


num_labels=4

python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \

python Delete_Excess.py


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train.py \
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















train_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Beer/Occur/train.json"
test_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Beer/Occur/anti-test.json"
val_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Beer/Occur/val.json"


num_labels=4

python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \

python Delete_Excess.py


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train.py \
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
















train_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Emotion/Auth/train.json"
test_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Emotion/Auth/test.json"
val_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Emotion/Auth/val.json"


num_labels=4

python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \

python Delete_Excess.py


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train.py \
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

















train_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Emotion/Auth/train.json"
test_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Emotion/Auth/anti-test.json"
val_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Emotion/Auth/val.json"


num_labels=4

python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \

python Delete_Excess.py


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train.py \
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
















train_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Emotion/Catg/train.json"
test_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Emotion/Catg/test.json"
val_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Emotion/Catg/val.json"


num_labels=4

python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \

python Delete_Excess.py


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train.py \
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
















train_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Emotion/Catg/train.json"
test_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Emotion/Catg/anti-test.json"
val_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Emotion/Catg/val.json"


num_labels=4

python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \

python Delete_Excess.py


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train.py \
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















train_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Emotion/Reg/train.json"
test_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Emotion/Reg/test.json"
val_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Emotion/Reg/val.json"


num_labels=4

python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \

python Delete_Excess.py


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train.py \
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



















train_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Emotion/Reg/train.json"
test_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Emotion/Reg/anti-test.json"
val_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Emotion/Reg/val.json"


num_labels=4

python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \

python Delete_Excess.py


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train.py \
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



















train_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Emotion/ST/train.json"
test_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Emotion/ST/test.json"
val_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Emotion/ST/val.json"


num_labels=4

python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \

python Delete_Excess.py


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train.py \
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

















train_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Emotion/ST/train.json"
test_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Emotion/ST/anti-test.json"
val_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Emotion/ST/val.json"


num_labels=4

python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \

python Delete_Excess.py


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train.py \
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



















train_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Emotion/Syn/train.json"
test_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Emotion/Syn/test.json"
val_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Emotion/Syn/val.json"


num_labels=4

python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \

python Delete_Excess.py


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train.py \
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






















train_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Emotion/Syn/train.json"
test_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Emotion/Syn/anti-test.json"
val_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Emotion/Syn/val.json"


num_labels=4

python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \

python Delete_Excess.py


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train.py \
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




















train_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Yelp/Auth/train.json"
test_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Yelp/Auth/test.json"
val_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Yelp/Auth/val.json"


num_labels=5

python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \

python Delete_Excess.py


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train.py \
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


















train_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Yelp/Auth/train.json"
test_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Yelp/Auth/anti-test.json"
val_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Yelp/Auth/val.json"


num_labels=5

python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \

python Delete_Excess.py


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train.py \
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






















train_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Yelp/Catg/train.json"
test_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Yelp/Catg/test.json"
val_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Yelp/Catg/val.json"


num_labels=5

python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \

python Delete_Excess.py


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train.py \
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




















train_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Yelp/Catg/train.json"
test_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Yelp/Catg/anti-test.json"
val_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Yelp/Catg/val.json"


num_labels=5

python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \

python Delete_Excess.py


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train.py \
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






















train_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Yelp/Reg/train.json"
test_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Yelp/Reg/test.json"
val_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Yelp/Reg/val.json"


num_labels=5

python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \

python Delete_Excess.py


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train.py \
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





















train_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Yelp/Reg/train.json"
test_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Yelp/Reg/anti-test.json"
val_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Yelp/Reg/val.json"


num_labels=5

python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \

python Delete_Excess.py


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train.py \
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























train_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Yelp/ST/train.json"
test_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Yelp/ST/test.json"
val_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Yelp/ST/val.json"


num_labels=5

python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \

python Delete_Excess.py


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train.py \
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






















train_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Yelp/ST/train.json"
test_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Yelp/ST/anti-test.json"
val_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Yelp/ST/val.json"


num_labels=5

python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \

python Delete_Excess.py


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train.py \
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

























train_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Yelp/Syn/train.json"
test_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Yelp/Syn/test.json"
val_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Yelp/Syn/val.json"


num_labels=5

python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \

python Delete_Excess.py


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train.py \
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



























train_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Yelp/Syn/train.json"
test_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Yelp/Syn/anti-test.json"
val_dataset="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_4_Delete_Directly/Shortcuts_Maze_with_Val/Yelp/Syn/val.json"


num_labels=5

python save_CC.py \
    --dataset_name "$train_dataset" \
    --model_name "$model_name" \

python Delete_Excess.py


for seed in ${seeds[@]}; do
    for strech_proportion in "${strech_proportions[@]}"; do
        echo "当前参数: strech_proportion=$strech_proportion"
        python Train.py \
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









python Make_F.py