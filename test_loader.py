from data.loader import data_loader
import argparse
import os

os.makedirs('./data/datasets/sdd_test', exist_ok=True)
os.system('cp ./data/datasets/sdd/bookstore_video0.txt ./data/datasets/sdd_test/')

args = argparse.Namespace(
    obs_len=8,
    pred_len=12,
    skip=1,
    delim='space',
    batch_size=4,
    loader_num_workers=0
)

print("Loading single file...")
dset, loader = data_loader(args, './data/datasets/sdd_test')
print(f"Dataset loaded! Total sequences: {len(dset)}")

batch = next(iter(loader))
print(f"Batch has {len(batch)} items (should be 8)")
print(f"Agent classes shape: {batch[7].shape}")
print(f"Sample classes: {batch[7][:10]}")
print(f"Unique classes: {batch[7].unique()}")