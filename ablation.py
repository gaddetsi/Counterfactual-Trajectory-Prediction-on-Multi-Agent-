import torch
import numpy as np
import json
import os
from collections import defaultdict

from data.loader import data_loader
from models import TrajectoryGenerator
from utils import relative_to_abs

print("ABLATION STUDY: Class-Conditioned Counterfactuals")
print("\nComparing:")
print("  1. Baseline: Uniform λ=0.5 for all agents")
print("  2. Extended: Class-specific λ (0.3 for pedestrians, 0.7 for vehicles)")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Data configuration
class Args:
    dataset_name = 'sdd_split'
    delim = ' '
    loader_num_workers = 4
    obs_len = 8
    pred_len = 12
    skip = 1
    batch_size = 8

args = Args()

CLASS_NAMES = ['Pedestrian', 'Biker', 'Skater', 'Cart', 'Car', 'Bus']

print("STEP 1: EVALUATE BASELINE (Uniform Lambda)")

# Load baseline model
print("\nLoading baseline model...")
baseline_checkpoint = torch.load('./CausalHTP/checkpoint/checkpoint0.pth.tar', map_location=device)

baseline_model = TrajectoryGenerator(
    obs_len=8, pred_len=12,
    traj_lstm_input_size=2, traj_lstm_hidden_size=32,
    n_units=[32, 16, 32], n_heads=[4, 1],
    graph_network_out_dims=32,
    dropout=0, alpha=0.2,
    graph_lstm_hidden_size=32,
    noise_dim=(16,), noise_type='gaussian'
)
baseline_model.load_state_dict(baseline_checkpoint['state_dict'])
baseline_model.to(device)
baseline_model.eval()
print("Baseline model loaded")

# Load test data
print("\nLoading test data...")
test_path = './data/datasets/sdd_split/test'
_, test_loader = data_loader(args, test_path)
print(f"Test data loaded: {len(test_loader)} batches")

# Evaluate baseline
print("\nEvaluating baseline model...")
baseline_metrics = {i: {'ade': [], 'fde': []} for i in range(6)}

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, \
        non_linear_ped, loss_mask, seq_start_end, agent_class_ids = batch
        
        # Forward pass
        dummy = torch.zeros_like(agent_class_ids)
        pred_traj_fake_rel = baseline_model(obs_traj_rel, obs_traj, seq_start_end, dummy)
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
        
        # Compute per-agent metrics
        for i in range(obs_traj.shape[1]):
            agent_class = int(agent_class_ids[i].item())
            
            ade = torch.norm(pred_traj_fake[:, i, :] - pred_traj_gt[:, i, :], dim=1).mean().item()
            fde = torch.norm(pred_traj_fake[-1, i, :] - pred_traj_gt[-1, i, :]).item()
            
            baseline_metrics[agent_class]['ade'].append(ade)
            baseline_metrics[agent_class]['fde'].append(fde)
        
        if (batch_idx + 1) % 2000 == 0:
            print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches...")

print("Baseline evaluation complete")

# Print baseline results
print("BASELINE RESULTS (Uniform Lambda)")
print(f"{'Class':<15} {'Count':<10} {'ADE':<12} {'FDE':<12}")

baseline_results = {}
for class_id in range(6):
    if baseline_metrics[class_id]['ade']:
        avg_ade = np.mean(baseline_metrics[class_id]['ade'])
        avg_fde = np.mean(baseline_metrics[class_id]['fde'])
        count = len(baseline_metrics[class_id]['ade'])
        baseline_results[CLASS_NAMES[class_id]] = {'ade': avg_ade, 'fde': avg_fde, 'count': count}
        print(f"{CLASS_NAMES[class_id]:<15} {count:<10} {avg_ade:<12.3f} {avg_fde:<12.3f}")

all_baseline_ade = [ade for m in baseline_metrics.values() for ade in m['ade']]
all_baseline_fde = [fde for m in baseline_metrics.values() for fde in m['fde']]
baseline_overall_ade = np.mean(all_baseline_ade)
baseline_overall_fde = np.mean(all_baseline_fde)

print(f"{'OVERALL':<15} {len(all_baseline_ade):<10} {baseline_overall_ade:<12.3f} {baseline_overall_fde:<12.3f}")

baseline_results['overall'] = {'ade': baseline_overall_ade, 'fde': baseline_overall_fde}

# STEP 2: Evaluate Extended Model
print("STEP 2: EVALUATE EXTENDED (Class-Conditioned Lambda)")

print("\nLoading extended model...")
extended_checkpoint = torch.load('./model_best.pth.tar', map_location=device)

extended_model = TrajectoryGenerator(
    obs_len=8, pred_len=12,
    traj_lstm_input_size=2, traj_lstm_hidden_size=32,
    n_units=[32, 16, 32], n_heads=[4, 1],
    graph_network_out_dims=32,
    dropout=0, alpha=0.2,
    graph_lstm_hidden_size=32,
    noise_dim=(16,), noise_type='gaussian'
)
extended_model.load_state_dict(extended_checkpoint['state_dict'])
extended_model.to(device)
extended_model.eval()
print("Extended model loaded")

# Check lambda values
print("\nClass-specific lambda values:")
lambda_values = extended_model.lambda_per_class.cpu().detach().numpy()
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name:<15} λ = {lambda_values[i]:.3f}")

# Evaluate extended
print("\nEvaluating extended model...")
extended_metrics = {i: {'ade': [], 'fde': []} for i in range(6)}

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, \
        non_linear_ped, loss_mask, seq_start_end, agent_class_ids = batch
        
        # Forward pass with agent class IDs
        pred_traj_fake_rel = extended_model(obs_traj_rel, obs_traj, seq_start_end, agent_class_ids)
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
        
        # Compute per-agent metrics
        for i in range(obs_traj.shape[1]):
            agent_class = int(agent_class_ids[i].item())
            
            ade = torch.norm(pred_traj_fake[:, i, :] - pred_traj_gt[:, i, :], dim=1).mean().item()
            fde = torch.norm(pred_traj_fake[-1, i, :] - pred_traj_gt[-1, i, :]).item()
            
            extended_metrics[agent_class]['ade'].append(ade)
            extended_metrics[agent_class]['fde'].append(fde)
        
        if (batch_idx + 1) % 2000 == 0:
            print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches...")

print("Extended evaluation complete")

# Print extended results
print("EXTENDED RESULTS (Class-Conditioned Lambda)")
print(f"{'Class':<15} {'Count':<10} {'ADE':<12} {'FDE':<12}")

extended_results = {}
for class_id in range(6):
    if extended_metrics[class_id]['ade']:
        avg_ade = np.mean(extended_metrics[class_id]['ade'])
        avg_fde = np.mean(extended_metrics[class_id]['fde'])
        count = len(extended_metrics[class_id]['ade'])
        extended_results[CLASS_NAMES[class_id]] = {'ade': avg_ade, 'fde': avg_fde, 'count': count}
        print(f"{CLASS_NAMES[class_id]:<15} {count:<10} {avg_ade:<12.3f} {avg_fde:<12.3f}")

all_extended_ade = [ade for m in extended_metrics.values() for ade in m['ade']]
all_extended_fde = [fde for m in extended_metrics.values() for fde in m['fde']]
extended_overall_ade = np.mean(all_extended_ade)
extended_overall_fde = np.mean(all_extended_fde)

print(f"{'OVERALL':<15} {len(all_extended_ade):<10} {extended_overall_ade:<12.3f} {extended_overall_fde:<12.3f}")

extended_results['overall'] = {'ade': extended_overall_ade, 'fde': extended_overall_fde}

# STEP 3: Comparison
print("ABLATION COMPARISON: Impact of Class-Conditioned Counterfactuals")
print(f"\n{'Class':<15} {'Baseline ADE':<15} {'Extended ADE':<15} {'Improvement':<15}")

improvements = {}
for class_name in CLASS_NAMES:
    if class_name in baseline_results and class_name in extended_results:
        base_ade = baseline_results[class_name]['ade']
        ext_ade = extended_results[class_name]['ade']
        improvement = ((base_ade - ext_ade) / base_ade) * 100
        improvements[class_name] = improvement
        
        print(f"{class_name:<15} {base_ade:<15.3f} {ext_ade:<15.3f} {improvement:>+13.1f}%")

# Overall improvement
overall_improvement = ((baseline_overall_ade - extended_overall_ade) / baseline_overall_ade) * 100
improvements['overall'] = overall_improvement

print(f"{'OVERALL':<15} {baseline_overall_ade:<15.3f} {extended_overall_ade:<15.3f} {overall_improvement:>+13.1f}%")

# Save results
print("\nSaving results...")
os.makedirs('./ablation', exist_ok=True)

ablation_results = {
    'baseline': baseline_results,
    'extended': extended_results,
    'improvements': improvements,
    'lambda_values': {CLASS_NAMES[i]: float(lambda_values[i]) for i in range(6)}
}

with open('./ablation/class_conditioned_ablation.json', 'w') as f:
    json.dump(ablation_results, f, indent=2)




print("\n2. Per-Class Improvements:")
for class_name in CLASS_NAMES:
    if class_name in improvements:
        print(f"   - {class_name:<12}: {improvements[class_name]:+6.1f}%")

print("\n3. Class-Specific Lambda Values:")
for i, name in enumerate(CLASS_NAMES):
    print(f"   - {name:<12}: λ = {lambda_values[i]:.3f}")

# Identify best/worst improvements
class_improvements = [(name, improvements[name]) for name in CLASS_NAMES if name in improvements]
class_improvements.sort(key=lambda x: x[1], reverse=True)

print("\n4. Ranked by Improvement:")
for rank, (name, imp) in enumerate(class_improvements, 1):
    print(f"   {rank}. {name:<12}: {imp:+6.1f}%")

