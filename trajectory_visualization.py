import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.loader import data_loader
from models import TrajectoryGenerator
from utils import relative_to_abs


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 0,  # Remove axis border
    "grid.alpha": 0.15
})


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Args:
    dataset_name = 'sdd_split'
    delim = ' '
    loader_num_workers = 0
    obs_len = 8
    pred_len = 12
    skip = 1
    batch_size = 32

args = Args()

CLASS_NAMES = {
    0: 'Pedestrian',
    1: 'Biker',
    2: 'Skater',
    3: 'Cart',
    4: 'Car',
    5: 'Bus'
}

TARGET_CLASSES = [0, 1, 4]

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TrajectoryGenerator(
        obs_len=8,
        pred_len=12,
        traj_lstm_input_size=2,
        traj_lstm_hidden_size=32,
        n_units=[32, 16, 32],
        n_heads=[4, 1],
        graph_network_out_dims=32,
        dropout=0,
        alpha=0.2,
        graph_lstm_hidden_size=32,
        noise_dim=(16,),
        noise_type='gaussian'
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model

print("Loading models...")
baseline_model = load_model('./CausalHTP/checkpoint/checkpoint0.pth.tar')
extended_model = load_model('./model_best.pth.tar')
print("Models loaded")


print("Loading test data...")
test_path = './data/datasets/sdd_split/test'
_, test_loader = data_loader(args, test_path)
print("✓ Data loaded")


print("Searching for visually clear examples...")

 
candidates = {c: [] for c in CLASS_NAMES.keys()}

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        # Stop if we have enough examples for all target classes
        if all(len(candidates[c]) >= 5 for c in TARGET_CLASSES):
            break

        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, \
        non_linear_ped, loss_mask, seq_start_end, agent_class_ids = batch

        # Average predictions (5 samples)
        baseline_preds = []
        extended_preds = []

        for _ in range(5):
            dummy = torch.zeros_like(agent_class_ids)
            pred_b = baseline_model(obs_traj_rel, obs_traj, seq_start_end, dummy)
            baseline_preds.append(relative_to_abs(pred_b, obs_traj[-1]))

            pred_e = extended_model(obs_traj_rel, obs_traj, seq_start_end, agent_class_ids)
            extended_preds.append(relative_to_abs(pred_e, obs_traj[-1]))

        pred_baseline = torch.stack(baseline_preds).mean(dim=0)
        pred_extended = torch.stack(extended_preds).mean(dim=0)

        for idx in range(obs_traj.shape[1]):
            cls = int(agent_class_ids[idx].item())
            
            # Skip if not in our class names or already have enough
            if cls not in CLASS_NAMES:
                continue
            if cls in TARGET_CLASSES and len(candidates[cls]) >= 5:
                continue

            obs = obs_traj[:, idx].cpu().numpy()
            gt = pred_traj_gt[:, idx].cpu().numpy()
            base = pred_baseline[:, idx].cpu().numpy()
            ext = pred_extended[:, idx].cpu().numpy()

            total_dist = np.linalg.norm(gt[-1] - obs[0])
            base_ade = np.linalg.norm(base - gt, axis=1).mean()
            ext_ade = np.linalg.norm(ext - gt, axis=1).mean()
            improvement = base_ade - ext_ade
            separation = np.linalg.norm(base - ext, axis=1).max()

            if total_dist > 50 and improvement > 0 and separation > 10:
                candidates[cls].append({
                    "obs": obs,
                    "gt": gt,
                    "baseline": base,
                    "extended": ext,
                    "name": CLASS_NAMES[cls],
                    "score": improvement * separation
                })


examples = []
used_classes = set()

# First pass: select best example from each target class
for cls in TARGET_CLASSES:
    if candidates[cls]:
        candidates[cls].sort(key=lambda x: x["score"], reverse=True)
        examples.append(candidates[cls][0])
        used_classes.add(cls)

# Second pass: if we still need more examples, use other classes
if len(examples) < 4:
    all_classes = list(CLASS_NAMES.keys())
    remaining_classes = [c for c in all_classes if c not in used_classes]
    
    for cls in remaining_classes:
        if len(examples) >= 4:
            break
        if candidates[cls]:
            candidates[cls].sort(key=lambda x: x["score"], reverse=True)
            examples.append(candidates[cls][0])
            used_classes.add(cls)

# If still not enough, relax the criteria and search more batches
if len(examples) < 4:
    print(f"Only found {len(examples)} distinct classes, relaxing criteria...")
    
    # Reset and search with relaxed criteria
    candidates = {c: [] for c in CLASS_NAMES.keys()}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if len(examples) >= 4:
                break
                
            batch = [tensor.to(device) for tensor in batch]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, \
            non_linear_ped, loss_mask, seq_start_end, agent_class_ids = batch

            baseline_preds = []
            extended_preds = []

            for _ in range(5):
                dummy = torch.zeros_like(agent_class_ids)
                pred_b = baseline_model(obs_traj_rel, obs_traj, seq_start_end, dummy)
                baseline_preds.append(relative_to_abs(pred_b, obs_traj[-1]))

                pred_e = extended_model(obs_traj_rel, obs_traj, seq_start_end, agent_class_ids)
                extended_preds.append(relative_to_abs(pred_e, obs_traj[-1]))

            pred_baseline = torch.stack(baseline_preds).mean(dim=0)
            pred_extended = torch.stack(extended_preds).mean(dim=0)

            for idx in range(obs_traj.shape[1]):
                cls = int(agent_class_ids[idx].item())
                if cls in used_classes:
                    continue

                obs = obs_traj[:, idx].cpu().numpy()
                gt = pred_traj_gt[:, idx].cpu().numpy()
                base = pred_baseline[:, idx].cpu().numpy()
                ext = pred_extended[:, idx].cpu().numpy()

                total_dist = np.linalg.norm(gt[-1] - obs[0])
                base_ade = np.linalg.norm(base - gt, axis=1).mean()
                ext_ade = np.linalg.norm(ext - gt, axis=1).mean()
                improvement = base_ade - ext_ade
                separation = np.linalg.norm(base - ext, axis=1).max()

                # Relaxed criteria
                if total_dist > 30 and separation > 5:
                    examples.append({
                        "obs": obs,
                        "gt": gt,
                        "baseline": base,
                        "extended": ext,
                        "name": CLASS_NAMES[cls],
                        "score": improvement * separation
                    })
                    used_classes.add(cls)
                    break

print(f"Found {len(examples)} examples from distinct classes: {[ex['name'] for ex in examples]}")


print("Generating visualization...")

# Adjust figure based on number of examples found
num_plots = len(examples)
fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4))
fig.patch.set_facecolor('#f2f2f7')

# Handle single plot case
if num_plots == 1:
    axes = [axes]

for i, (ax, ex) in enumerate(zip(axes, examples)):
    obs = ex["obs"]
    gt = ex["gt"]
    base = ex["baseline"]
    ext = ex["extended"]

    ax.set_facecolor('#f2f2f7')

    # Trajectories
    ax.plot(obs[:, 0], obs[:, 1],
            color='#d32f2f', lw=2.8, label='Observed')

    ax.plot(gt[:, 0], gt[:, 1],
            color='#1565c0', lw=2.8, label='Ground Truth')

    ax.plot(ext[:, 0], ext[:, 1],
            '--', color='#00acc1', lw=2.6, label='Ours')

    ax.plot(base[:, 0], base[:, 1],
            '--', color='#fbc02d', lw=2.6, label='Baseline')

    # End points
    ax.scatter(gt[-1, 0], gt[-1, 1], color='#1565c0', s=30, zorder=5)
    ax.scatter(ext[-1, 0], ext[-1, 1], color='#00acc1', s=30, zorder=5)
    ax.scatter(base[-1, 0], base[-1, 1], color='#fbc02d', s=30, zorder=5)

    # Formatting - NO BOUNDARIES
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True)
    
    # Remove spines (boundaries)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(f"({chr(97+i)}) {ex['name']}", fontsize=12)

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center',
           ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.05))

fig.suptitle("Baseline vs. Ours (Trajectory Prediction)",
             fontsize=15, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("trajectories.png", dpi=300, bbox_inches='tight')
plt.close()

