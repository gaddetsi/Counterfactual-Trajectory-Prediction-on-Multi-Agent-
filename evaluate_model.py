import argparse
import os

import torch
import numpy as np

from data.loader import data_loader
from models import TrajectoryGenerator
from utils import (
    displacement_error,
    final_displacement_error,
    l2_loss,
    int_tuple,
    relative_to_abs,
    get_dset_path,
)
import time

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", default="./", help="Directory containing logging file")

parser.add_argument("--dataset_name", default="zara2", type=str)
parser.add_argument("--delim", default="\t")
parser.add_argument("--loader_num_workers", default=4, type=int)
parser.add_argument("--obs_len", default=8, type=int)
parser.add_argument("--pred_len", default=12, type=int)
parser.add_argument("--skip", default=1, type=int)

parser.add_argument("--seed", type=int, default=72, help="Random seed.")
parser.add_argument("--batch_size", default=64, type=int)

parser.add_argument("--noise_dim", default=(16,), type=int_tuple)
parser.add_argument("--noise_type", default="gaussian")
parser.add_argument("--noise_mix_type", default="global")

parser.add_argument(
    "--traj_lstm_input_size", type=int, default=2, help="traj_lstm_input_size"
)
parser.add_argument("--traj_lstm_hidden_size", default=32, type=int)

parser.add_argument(
    "--heads", type=str, default="4,1", help="Heads in each layer, splitted with comma"
)
parser.add_argument(
    "--hidden-units",
    type=str,
    default="16",
    help="Hidden units in each hidden layer, splitted with comma",
)
parser.add_argument(
    "--graph_network_out_dims",
    type=int,
    default=32,
    help="dims of every node after through GAT module",
)
parser.add_argument("--graph_lstm_hidden_size", default=32, type=int)


parser.add_argument("--num_samples", default=20, type=int)


parser.add_argument(
    "--dropout", type=float, default=0, help="Dropout rate (1 - keep probability)."
)
parser.add_argument(
    "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
)

parser.add_argument("--dset_type", default="test", type=str)


parser.add_argument(
    "--resume",
    default="./model_best.pth.tar",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument("--gpu_num", default="0", type=str)

# NEW: Added option to save per-class metrics
parser.add_argument(
    "--save_per_class",
    action="store_true",
    help="Save detailed per-class metrics to file"
)

# Agent class names for reporting
CLASS_NAMES = {
    0: "Pedestrian",
    1: "Biker", 
    2: "Skater",
    3: "Cart",
    4: "Car",
    5: "Bus"
}


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)
    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def get_generator(checkpoint):
    n_units = (
        [args.traj_lstm_hidden_size]
        + [int(x) for x in args.hidden_units.strip().split(",")]
        + [args.graph_lstm_hidden_size]
    )
    n_heads = [int(x) for x in args.heads.strip().split(",")]
    model = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        traj_lstm_input_size=args.traj_lstm_input_size,
        traj_lstm_hidden_size=args.traj_lstm_hidden_size,
        n_units=n_units,
        n_heads=n_heads,
        graph_network_out_dims=args.graph_network_out_dims,
        dropout=args.dropout,
        alpha=args.alpha,
        graph_lstm_hidden_size=args.graph_lstm_hidden_size,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda()
    model.eval()
    return model


def cal_ade_fde(pred_traj_gt, pred_traj_fake):
    ade = displacement_error(pred_traj_fake, pred_traj_gt, mode="raw")
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode="raw")
    return ade, fde


def evaluate(args, loader, generator):
    ade_outer, fde_outer = [], []
    total_traj = 0
    alltime = 0
    step = 0
    
    # NEW: Track per-class metrics
    class_ade = {i: [] for i in range(6)}  # 6 agent classes
    class_fde = {i: [] for i in range(6)}
    class_counts = {i: 0 for i in range(6)}
    
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (
                obs_traj,
                pred_traj_gt,
                obs_traj_rel,
                pred_traj_gt_rel,
                non_linear_ped,
                loss_mask,
                seq_start_end,
                agent_class_ids,  # NEW: Added agent class IDs
            ) = batch
            
            step += seq_start_end.shape[0]
            start = time.time()
            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            for _ in range(args.num_samples):
                # MODIFIED: Pass agent_class_ids to generator
                pred_traj_fake_rel = generator(
                    obs_traj_rel, obs_traj, seq_start_end, agent_class_ids, 0, 3
                )
                pred_traj_fake_rel = pred_traj_fake_rel[-args.pred_len :]

                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                ade_, fde_ = cal_ade_fde(pred_traj_gt, pred_traj_fake)
                ade.append(ade_)
                fde.append(fde_)
            
            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
            
            # NEW: Compute per-class errors
            # Get best prediction across samples
            ade_tensor = torch.stack(ade, dim=1)  # [num_agents, num_samples]
            fde_tensor = torch.stack(fde, dim=1)  # [num_agents, num_samples]
            
            # For each agent, get minimum error across samples
            ade_min, _ = ade_tensor.min(dim=1)  # [num_agents]
            fde_min, _ = fde_tensor.min(dim=1)  # [num_agents]
            
            # Normalize ADE by prediction length (divide by pred_len)
            ade_min = ade_min / args.pred_len
            
            # Accumulate per class
            for i, class_id in enumerate(agent_class_ids):
                class_id = class_id.item()
                class_ade[class_id].append(ade_min[i].item())
                class_fde[class_id].append(fde_min[i].item())
                class_counts[class_id] += 1
            
            elapsed = (time.time() - start)
            alltime += elapsed
        
        # Compute overall metrics
        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        
        # NEW: Compute per-class metrics
        class_metrics = {}
        for class_id in range(6):
            if class_counts[class_id] > 0:
                class_metrics[class_id] = {
                    'ade': np.mean(class_ade[class_id]),
                    'fde': np.mean(class_fde[class_id]),
                    'count': class_counts[class_id]
                }
        
        return ade, fde, class_metrics


def print_class_metrics(class_metrics):
    """Print per-class metrics in a formatted table"""
    print("PER-CLASS METRICS")
    print(f"{'Class':<15} {'Count':<10} {'ADE':<15} {'FDE':<15}")
    
    for class_id in sorted(class_metrics.keys()):
        metrics = class_metrics[class_id]
        class_name = CLASS_NAMES.get(class_id, f"Class_{class_id}")
        print(f"{class_name:<15} {metrics['count']:<10} "
              f"{metrics['ade']:<15.6f} {metrics['fde']:<15.6f}")
    


def save_class_metrics(class_metrics, dataset_name, output_file="class_metrics.txt"):
    """Save per-class metrics to file"""
    with open(output_file, 'w') as f:
        f.write(f"Per-Class Metrics - {dataset_name}\n")
        f.write(f"{'Class':<15} {'Count':<10} {'ADE':<15} {'FDE':<15}\n")
        
        for class_id in sorted(class_metrics.keys()):
            metrics = class_metrics[class_id]
            class_name = CLASS_NAMES.get(class_id, f"Class_{class_id}")
            f.write(f"{class_name:<15} {metrics['count']:<10} "
                   f"{metrics['ade']:<15.6f} {metrics['fde']:<15.6f}\n")
        
    
    print(f"\n✓ Per-class metrics saved to {output_file}")


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    checkpoint = torch.load(args.resume)
    generator = get_generator(checkpoint)
    path = "./data/datasets/sdd_split/test"

    _, loader = data_loader(args, path)
    
    # MODIFIED: Now returns per-class metrics too
    ade, fde, class_metrics = evaluate(args, loader, generator)
    
    # Print overall metrics
    print("\n" + "=" * 80)
    print("OVERALL METRICS")
    print(
        "Dataset: {}, Pred Len: {}, ADE: {:.6f}, FDE: {:.6f}".format(
            args.dataset_name, args.pred_len, ade, fde
        )
    )
    
    # Print per-class metrics
    print_class_metrics(class_metrics)
    
    # Save if requested
    if args.save_per_class:
        output_file = f"{args.dataset_name}_class_metrics.txt"
        save_class_metrics(class_metrics, args.dataset_name, output_file)


if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(72)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main(args)