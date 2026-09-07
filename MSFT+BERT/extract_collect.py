import pandas as pd
import argparse
import os
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset_name', type=str, default='train_61111_revised', help='dataset name')

args = parser.parse_args()
dataset_name = args.dataset_name

df_ls = []
dir_files = os.listdir(f'results/{dataset_name}')
for file in dir_files:
    df = pd.read_csv(f'results/{dataset_name}/{file}')
    df_ls.append(df)
df = pd.concat(df_ls)
df.to_csv(f'results/{dataset_name}/total.csv', index=False)
print('='*80)
print(f'extracted {len(df)} results from {len(dir_files)} files')
print(f'saved to results/{dataset_name}/total.csv')