# train_drift_agent2.py (Upgraded with fallback handling and fixed autograd error)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from drift_env import TransactionRoutingEnv
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data as GraphData
from collections import deque
import random
import networkx as nx

class GNNPolicy(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.actor = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        logits = self.actor(x)
        return logits

def graph_to_data(graph):
    node_features = []
    for node in graph.nodes():
        node_features.append([graph.nodes[node].get("balance", 1.0)])

    edge_index = []
    for u, v in graph.edges():
        edge_index.append([u, v])
        edge_index.append([v, u])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_features, dtype=torch.float)
    return GraphData(x=x, edge_index=edge_index)

def mask_invalid_actions(graph, current_node, num_nodes):
    valid_mask = torch.zeros(num_nodes)
    for neighbor in graph.neighbors(current_node):
        if not graph[current_node][neighbor].get("regulatory_blocked", False):
            valid_mask[neighbor] = 1
    return valid_mask

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TransactionRoutingEnv()
    model = GNNPolicy(in_channels=1, hidden_channels=32, out_channels=env.num_nodes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    gamma = 0.99
    max_episodes = 50000
    episode_rewards = []

    for ep in range(max_episodes):
        obs, _ = env.reset()
        graph_data = graph_to_data(env.graph).to(device)
        current_node = obs['current_node']
        destination = obs['destination']

        log_probs, rewards = [], []
        done = False
        total_reward = 0

        while not done:
            logits = model(graph_data)
            probs = F.softmax(logits, dim=-1)[current_node]
            mask = mask_invalid_actions(env.graph, current_node, env.num_nodes).to(device)
            masked_probs = probs * mask

            if masked_probs.sum().item() <= 1e-6:
                valid_neighbors = [nbr for nbr in env.graph.neighbors(current_node)
                                   if not env.graph[current_node][nbr].get("regulatory_blocked", False)]
                if not valid_neighbors:
                    action = random.randint(0, env.num_nodes - 1)
                else:
                    action = random.choice(valid_neighbors)
                prob_log = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                masked_probs = masked_probs / masked_probs.sum()
                dist = torch.distributions.Categorical(masked_probs)
                action_tensor = torch.tensor(action := dist.sample().item()).to(device)
                prob_log = dist.log_prob(action_tensor)

            obs, reward, term, trunc, _ = env.step(action)

            edge = env.graph.get_edge_data(current_node, action)
            if edge:
                reward -= 0.01 * edge['fee']
                reward += 0.02 * edge.get('reliability', 0.5)

            current_node = obs['current_node']
            log_probs.append(prob_log)
            rewards.append(reward)
            total_reward += reward
            done = term or trunc

        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(device)
        if returns.std().item() > 1e-6:
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        policy_loss = [-log_prob * R for log_prob, R in zip(log_probs, returns)]
        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        optimizer.step()

        episode_rewards.append(total_reward)
        if ep % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {ep}, Average Reward: {avg_reward:.2f}")

    torch.save(model.state_dict(), "drift_policy.pt")
    print("✅ Model training complete and saved.")

if __name__ == '__main__':
    train()
