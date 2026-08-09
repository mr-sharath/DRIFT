# train_waynex.py

"""
Offline Pre-Training Script for Waynex Neural Engine.
Trains the Waynex GNN + Actor-Critic Policy on diverse synthetic and real city graph topologies.
Saves model weights to `waynex_policy.pt`.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import os
from drl_policy import WaynexActorCriticPolicy, prepare_graph_node_features, create_fully_connected_edge_index
from osrm_client import get_haversine_matrix


def train_policy(epochs: int = 150, checkpoint_path: str = "waynex_policy.pt"):
    print(f"🚀 Starting Waynex Neural Policy Training ({epochs} epochs)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WaynexActorCriticPolicy(in_features=6, hidden_dim=64, embed_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    
    for epoch in range(1, epochs + 1):
        # Generate random city graph episode (1 depot + 8..15 delivery nodes)
        num_deliveries = random.randint(8, 15)
        num_nodes = num_deliveries + 1
        
        # Base center (Bengaluru lat/lng)
        base_lat, base_lng = 12.9716, 77.5946
        coords = [{"lat": base_lat, "lng": base_lng}]
        
        deliveries = []
        for i in range(num_deliveries):
            d_lat = base_lat + random.uniform(-0.08, 0.08)
            d_lng = base_lng + random.uniform(-0.08, 0.08)
            coords.append({"lat": d_lat, "lng": d_lng})
            deliveries.append({"demand": random.randint(50, 300), "time_window": [9, 17]})

        dist_mat, dur_mat = get_haversine_matrix(coords)
        node_features = prepare_graph_node_features(coords, deliveries).to(device)
        edge_index = create_fully_connected_edge_index(num_nodes).to(device)

        # Simulate routing for 2 vehicles
        optimizer.zero_grad()
        total_loss = 0.0

        unvisited = set(range(1, num_nodes))
        v_cap = 600

        for _ in range(2):
            curr_cap = 0
            curr_node = 0
            curr_dur = 0.0

            while unvisited:
                veh_state = torch.tensor([[curr_cap / v_cap, 1.0, curr_dur / 300.0, 1.0]], dtype=torch.float32).to(device)
                action_mask = torch.zeros(num_nodes, dtype=torch.float32).to(device)
                
                valid_candidates = []
                for cand in unvisited:
                    demand = deliveries[cand - 1]["demand"]
                    if curr_cap + demand <= v_cap:
                        action_mask[cand] = 1.0
                        valid_candidates.append(cand)

                if not valid_candidates:
                    break

                logits, masked_logits, probs, value = model(node_features, edge_index, curr_node, veh_state, action_mask)
                
                # Sample action
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                # Compute step reward: -distance + success bonus
                chosen_node = action.item()
                step_dist = dist_mat[curr_node][chosen_node]
                reward = -step_dist + 5.0

                # Advantage & Loss
                advantage = torch.tensor([reward], device=device) - value
                actor_loss = -log_prob * advantage.detach()
                critic_loss = F.mse_loss(value, torch.tensor([reward], device=device))
                step_loss = actor_loss + 0.5 * critic_loss

                step_loss.backward()
                total_loss += step_loss.item()

                unvisited.remove(chosen_node)
                curr_cap += deliveries[chosen_node - 1]["demand"]
                curr_dur += dur_mat[curr_node][chosen_node]
                curr_node = chosen_node

        optimizer.step()

        if epoch % 30 == 0 or epoch == epochs:
            print(f" [Epoch {epoch}/{epochs}] Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), checkpoint_path)
    print(f"✅ Saved trained policy checkpoint to `{checkpoint_path}`")


if __name__ == "__main__":
    train_policy(epochs=100)
