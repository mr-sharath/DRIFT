import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data as GraphData
import numpy as np
import random
from drift_env import TransactionRoutingEnv
from train_drift_agent2 import graph_to_data, mask_invalid_actions

class GNNActorCritic(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.actor_head = nn.Linear(hidden_channels, out_channels)
        self.critic_head = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        logits = self.actor_head(x)
        values = self.critic_head(x).squeeze(-1)
        return logits, values

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TransactionRoutingEnv()
    model = GNNActorCritic(in_channels=2, hidden_channels=64, out_channels=env.num_nodes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    gamma = 0.99
    max_episodes = 10000
    episode_rewards = []

    for ep in range(max_episodes):
        obs, _ = env.reset()
        current_node = obs['current_node']
        destination = obs['destination']

        # Node features: [is_current_node, is_destination]
        x = []
        for i in range(env.num_nodes):
            x.append([
                1.0 if i == current_node else 0.0,
                1.0 if i == destination else 0.0
            ])
        graph_data = graph_to_data(env.graph).to(device)
        graph_data.x = torch.tensor(x, dtype=torch.float).to(device)

        log_probs, values, rewards = [], [], []
        total_reward = 0
        done = False

        while not done:
            logits, value_preds = model(graph_data)
            probs = F.softmax(logits, dim=-1)[current_node]
            mask = mask_invalid_actions(env.graph, current_node, env.num_nodes).to(device)
            masked_probs = probs * mask

            if masked_probs.sum().item() <= 1e-6:
                valid_neighbors = [nbr for nbr in env.graph.neighbors(current_node)
                                   if not env.graph[current_node][nbr].get("regulatory_blocked", False)]
                action = random.choice(valid_neighbors) if valid_neighbors else random.randint(0, env.num_nodes - 1)
                log_prob = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                masked_probs = masked_probs / masked_probs.sum()
                dist = torch.distributions.Categorical(masked_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            # Safely handle both torch.tensor and int cases
            obs, reward, term, trunc, _ = env.step(int(action))
            done = term or trunc

            current_node = obs['current_node']
            log_probs.append(log_prob)
            values.append(value_preds[current_node])
            rewards.append(reward)
            total_reward += reward

            # Create new node features tensor instead of modifying in-place
            new_features = []
            for i in range(env.num_nodes):
                new_features.append([
                    1.0 if i == current_node else 0.0,
                    1.0 if i == destination else 0.0
                ])
            graph_data.x = torch.tensor(new_features, dtype=torch.float).to(device)

        episode_rewards.append(total_reward)

        # Compute returns and advantage
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float).to(device)
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)

        advantage = returns - values.detach()
        policy_loss = -(log_probs * advantage).mean()
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"[Episode {ep}] Avg Reward (last 100): {avg_reward:.2f}")

    torch.save(model.state_dict(), "drift_actor_critic.pt")
    print("✅ Actor-Critic model saved as drift_actor_critic.pt")

if __name__ == "__main__":
    train()