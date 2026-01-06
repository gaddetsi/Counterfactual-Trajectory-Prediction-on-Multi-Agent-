# Multi-Agent Counterfactual Trajectory Prediction

Extension of CausalHTP to heterogeneous urban traffic scenarios with multiple agent types.
---

## Overview

This project extends counterfactual trajectory prediction to handle diverse agent types (pedestrians, cyclists, cars, buses) in urban traffic scenes. We introduce:

1. **Agent-Type Embeddings** - Learnable representations for different agent classes
2. **Class-Conditioned Counterfactuals** - Agent-specific bias correction (λ values)
3. **Spatial Context Features** - Local density and scene zones

**Dataset:** Stanford Drone Dataset (SDD)  
**Baseline:** CausalHTP (ICCV 2021)  
**Results:** +4.7% ADE, +5.9% FDE overall improvement

---

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.8+
- CUDA (optional, for GPU)

### Setup
```bash
# Clone repository
git clone https://github.com/gaddetsi/Counterfactual-Trajectory-Prediction-on-Multi-Agent-.git
cd CausalHTP

# Create environment
conda create -n causalhtp python=3.8
conda activate causalhtp

# Install dependencies
pip install torch torchvision
pip install numpy pandas matplotlib networkx tqdm tensorboard
```

---

## Dataset Preparation

### Download Stanford Drone Dataset
```bash
# Option 1: Kaggle (recommended, 1.5GB)
# Download from: https://www.kaggle.com/datasets/aryashah2k/stanford-drone-dataset

# Option 2: Official (66GB)
# Download from: http://cvgl.stanford.edu/projects/uav_data/
```

### Preprocess Data
```bash
python preprocess_sdd.py
```

**Output:** Creates `data/datasets/sdd_split/` with train/test splits

---

## Training

### Train Extended Model (Full)
```bash
python train.py \
  --dataset_name sdd_split \
  --delim space \
  --batch_size 8 \
  --num_epochs 10 \
  --best_k 5
```

**Training time:** ~18 hours (10 epochs on single GPU)  
**Output:** `checkpoint/model_best.pth.tar`

### Train Baseline
```bash
cd CausalHTP
python train.py \
  --dataset_name sdd_split \
  --delim space \
  --batch_size 8 \
  --num_epochs 10
```

**Output:** `/CASUAL-HTP-MULTI/model_best.pth.tar`

---

## Evaluation

### Evaluate Extended Model
```bash
python evaluate_model.py \
  --dataset_name sdd_split \
  --delim space \
  --resume ./model_best.pth.tar \
  --batch_size 8 \
  --num_samples 10
```

### Evaluate Baseline
```bash
python evaluate_model.py \
  --dataset_name sdd_split \
  --delim space \
  --resume ./CausalHTP/checkpoint/checkpoint0.pth.tar \
  --batch_size 8 \
  --num_samples 10
```

---

## Ablation Study

### Run Ablation (Class-Conditioned Only)
```bash
python ablation.py
```

**Time:** ~30-45 minutes (evaluation only)  
**Output:** `ablation/class_conditioned_ablation.json`

---

## Visualization

### Generate Trajectory Comparisons
```bash
python trajectory_visualization.py
```

**Output:** `trajectories.png` 


## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python train.py --batch_size 4
```

### CUDA Not Available
```bash
# Use CPU (slower)
python train.py --device cpu
```

### Missing Dependencies
```bash
pip install --break-system-packages <package_name>
```

---

## Citation

If you use this code, please cite the original CausalHTP paper:
```bibtex
@inproceedings{chen2021human,
  title={Human trajectory prediction via counterfactual analysis},
  author={Chen, Guangyi and Li, Junlong and Lu, Jiwen and Zhou, Jie},
  booktitle={ICCV},
  year={2021}
}
```

---



## License

Based on CausalHTP (MIT License)
