import os
import shutil
from pathlib import Path
import random
import argparse


def split_sdd_data(sdd_processed_dir, output_base_dir, train_ratio=0.8, seed=42):
    """
    Split processed SDD files into train and test sets
    
    Args:
        sdd_processed_dir: Directory with processed SDD files (e.g., ./data/datasets/sdd)
        output_base_dir: Base directory for train/test split (e.g., ./data/datasets/sdd_split)
        train_ratio: Ratio of scenes for training (default 0.8)
        seed: Random seed for reproducibility
    """
    
    random.seed(seed)
    
    # Create output directories
    train_dir = Path(output_base_dir) / 'train'
    test_dir = Path(output_base_dir) / 'test'
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all processed files
    sdd_path = Path(sdd_processed_dir)
    all_files = sorted([f for f in sdd_path.glob('*.txt') 
                       if f.name != 'processing_summary.txt'])
    
    if not all_files:
        print(f"ERROR: No .txt files found in {sdd_processed_dir}")
        return
    
    print(f"Found {len(all_files)} scene files")
    print(f"Split ratio: {train_ratio:.0%} train, {1-train_ratio:.0%} test")
    
    # Group files by scene (before the first underscore)
    # e.g., "bookstore_video0.txt" -> scene "bookstore"
    scenes = {}
    for f in all_files:
        scene_name = f.stem.split('_')[0]  # Get scene name before first underscore
        if scene_name not in scenes:
            scenes[scene_name] = []
        scenes[scene_name].append(f)
    
    print(f"\nFound {len(scenes)} unique scenes:")
    for scene_name, files in sorted(scenes.items()):
        print(f"  {scene_name}: {len(files)} videos")
    
    # Shuffle scenes
    scene_names = list(scenes.keys())
    random.shuffle(scene_names)
    
    # Split scenes
    num_train_scenes = int(len(scene_names) * train_ratio)
    train_scenes = scene_names[:num_train_scenes]
    test_scenes = scene_names[num_train_scenes:]
    
    print("TRAIN SCENES:")
    train_files_count = 0
    for scene in sorted(train_scenes):
        print(f"  {scene}: {len(scenes[scene])} videos")
        train_files_count += len(scenes[scene])
    
    print("TEST SCENES:")
    test_files_count = 0
    for scene in sorted(test_scenes):
        print(f"  {scene}: {len(scenes[scene])} videos")
        test_files_count += len(scenes[scene])
    
    # Copy files to train/test directories
    print("Copying files...")
    
    for scene in train_scenes:
        for file in scenes[scene]:
            dest = train_dir / file.name
            shutil.copy2(file, dest)
    print(f"Copied {train_files_count} files to {train_dir}")
    
    for scene in test_scenes:
        for file in scenes[scene]:
            dest = test_dir / file.name
            shutil.copy2(file, dest)
    print(f"Copied {test_files_count} files to {test_dir}")
    
    # Save split info
    split_info_file = Path(output_base_dir) / 'split_info.txt'
    with open(split_info_file, 'w') as f:
        f.write("SDD Train/Test Split\n")
        f.write(f"Random seed: {seed}\n")
        f.write(f"Train ratio: {train_ratio}\n\n")
        f.write(f"Train scenes: {sorted(train_scenes)}\n")
        f.write(f"Test scenes: {sorted(test_scenes)}\n\n")
        f.write(f"Total train files: {train_files_count}\n")
        f.write(f"Total test files: {test_files_count}\n")
    
    print(f"\n Split info saved to {split_info_file}")
    

def main():
    parser = argparse.ArgumentParser(description='Split SDD dataset into train/test')
    parser.add_argument(
        '--input_dir',
        type=str,
        default='./data/datasets/sdd',
        help='Directory with processed SDD files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data/datasets/sdd_split',
        help='Output directory for train/test split'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Ratio of scenes for training (default: 0.8)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    split_sdd_data(
        args.input_dir,
        args.output_dir,
        args.train_ratio,
        args.seed
    )


if __name__ == '__main__':
    main()