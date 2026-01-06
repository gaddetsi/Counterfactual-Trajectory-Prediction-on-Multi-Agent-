# test_model.py
import torch
from models import TrajectoryGenerator

# Create model
model = TrajectoryGenerator(
    obs_len=8,
    pred_len=12,
    traj_lstm_input_size=2,
    traj_lstm_hidden_size=32,
    n_units=[32, 16, 32],
    n_heads=[4, 1],
    graph_network_out_dims=32,
    dropout=0.0,
    alpha=0.2,
    graph_lstm_hidden_size=32,
    noise_dim=(16,),
    noise_type="gaussian",
)
model.cuda()

print("Model created successfully!")
print(f"Agent embedding: {model.agent_type_embedding}")
print(f"Lambda per class: {model.lambda_per_class}")
print(f"Env mean shape: {model.env_mean_per_class.shape}")