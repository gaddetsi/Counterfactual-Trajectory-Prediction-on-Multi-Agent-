

import os
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

"""
SDD Dataset Preprocessing Script
Converts Stanford Drone Dataset annotations to CausalHTP format

Input: annotations.txt with format:
  track_id xmin ymin xmax ymax frame lost occluded generated "label"

Output: Processed files with format:
  frame_id track_id x y class_id
"""

# Agent class mapping
LABEL_MAP = {
    'Pedestrian': 0,
    'Biker': 1,
    'Skater': 2,
    'Cart': 3,
    'Car': 4,
    'Bus': 5
}


def process_scene(annotation_file, output_file, scene_name):
    """
    Process a single scene's annotation file
    
    Args:
        annotation_file: Path to annotations.txt
        output_file: Path to output processed file
        scene_name: Name of the scene
    """
    print(f"Processing scene: {scene_name}")
    
    # Read annotation file
    # Format: track_id xmin ymin xmax ymax frame lost occluded generated label
    try:
        data = []
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 10:
                    continue
                
                track_id = int(parts[0])
                xmin = float(parts[1])
                ymin = float(parts[2])
                xmax = float(parts[3])
                ymax = float(parts[4])
                frame = int(parts[5])
                # Skip lost, occluded, generated flags
                label = parts[9].strip('"')  # Remove quotes
                
                # Convert bounding box to center point
                x_center = (xmin + xmax) / 2.0
                y_center = (ymin + ymax) / 2.0
                
                # Map label to class ID
                class_id = LABEL_MAP.get(label, 0)  # Default to Pedestrian if unknown
                
                data.append([frame, track_id, x_center, y_center, class_id])
        
        # Convert to numpy array
        data = np.array(data)
        print(f"  Total frames: {len(np.unique(data[:, 0]))}")
        print(f"  Total tracks: {len(np.unique(data[:, 1]))}")
        
        # Count per class
        for label, class_id in LABEL_MAP.items():
            count = np.sum(data[:, 4] == class_id)
            if count > 0:
                print(f"    {label}: {count} detections")
        
        # Save to output file in tab-delimited format
        # Format: frame_id track_id x y class_id
        np.savetxt(output_file, data, fmt='%d %d %.4f %.4f %d', delimiter='\t')
        print(f"  Saved to: {output_file}")
        print(f"  Total detections: {len(data)}\n")
        
        return len(data), len(np.unique(data[:, 1]))
        
    except Exception as e:
        print(f"  ERROR processing {scene_name}: {e}\n")
        return 0, 0


def process_all_scenes(sdd_root, output_dir):
    """
    Process all scenes in SDD dataset
    
    Args:
        sdd_root: Root directory of SDD dataset (contains annotations/)
        output_dir: Output directory for processed files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    annotations_dir = Path(sdd_root) / 'annotations'
    
    if not annotations_dir.exists():
        print(f"ERROR: annotations directory not found at {annotations_dir}")
        return
    
    # Get all scene directories
    scenes = [d for d in annotations_dir.iterdir() if d.is_dir()]
    
    if not scenes:
        print(f"ERROR: No scene directories found in {annotations_dir}")
        return
    
    print(f"Found {len(scenes)} scenes\n")
    
    total_detections = 0
    total_tracks = 0
    scene_stats = []
    
    # Process each scene
    for scene_dir in sorted(scenes):
        scene_name = scene_dir.name
        
        # Find all video subdirectories
        video_dirs = [v for v in scene_dir.iterdir() if v.is_dir()]
        
        for video_dir in video_dirs:
            video_name = video_dir.name
            annotation_file = video_dir / 'annotations.txt'
            
            if not annotation_file.exists():
                print(f"WARNING: No annotations.txt in {video_dir}")
                continue
            
            # Create output filename
            output_filename = f"{scene_name}_{video_name}.txt"
            output_file = Path(output_dir) / output_filename
            
            # Process this scene/video
            num_detections, num_tracks = process_scene(
                annotation_file, 
                output_file, 
                f"{scene_name}/{video_name}"
            )
            
            total_detections += num_detections
            total_tracks += num_tracks
            scene_stats.append({
                'scene': scene_name,
                'video': video_name,
                'detections': num_detections,
                'tracks': num_tracks
            })
    
    
    # Save summary statistics
    summary_file = Path(output_dir) / 'processing_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("Scene,Video,Detections,Tracks\n")
        for stat in scene_stats:
            f.write(f"{stat['scene']},{stat['video']},{stat['detections']},{stat['tracks']}\n")
    print(f"Summary saved to: {summary_file}")


def verify_processed_data(output_dir):
    """
    Verify processed data format
    
    Args:
        output_dir: Directory containing processed files
    """
    
    files = list(Path(output_dir).glob('*.txt'))
    files = [f for f in files if f.name != 'processing_summary.txt']
    
    if not files:
        print("No processed files found!")
        return
    
    # Check first file
    test_file = files[0]
    print(f"Checking format of: {test_file.name}")
    
    data = np.loadtxt(test_file)
    print(f"Shape: {data.shape}")
    print(f"Columns: frame_id, track_id, x, y, class_id")
    print(f"\nFirst 5 rows:")
    print(data[:5])
    
    # Check class distribution
    print(f"\nClass distribution:")
    unique_classes, counts = np.unique(data[:, 4].astype(int), return_counts=True)
    for class_id, count in zip(unique_classes, counts):
        label = [k for k, v in LABEL_MAP.items() if v == class_id][0]
        print(f"  {label} (ID={class_id}): {count}")
    


def main():
    parser = argparse.ArgumentParser(description='Preprocess SDD dataset')
    parser.add_argument(
        '--sdd_root', 
        type=str, 
        required=True,
        help='Root directory of SDD dataset (contains annotations/ folder)'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./data/datasets/sdd',
        help='Output directory for processed files'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify processed data after conversion'
    )
    
    args = parser.parse_args()
    
    print("SDD Dataset Preprocessing")
    print(f"Input directory: {args.sdd_root}")
    print(f"Output directory: {args.output_dir}")
    
    # Process all scenes
    process_all_scenes(args.sdd_root, args.output_dir)
    
    # Verify if requested
    if args.verify:
        verify_processed_data(args.output_dir)


if __name__ == '__main__':
    main()