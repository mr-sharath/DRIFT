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
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, SAGEConv):
                nn.init.xavier_uniform_(m)

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
    
    # Use RMSprop optimizer with momentum
    optimizer = optim.RMSprop(model.parameters(), lr=3e-4, alpha=0.99, eps=1e-5)

    gamma = 0.99
    lam = 0.95  # GAE lambda
    max_episodes = 3000
    episode_rewards = []

    for ep in range(max_episodes):
        obs, _ = env.reset()
        current_node = obs['current_node']
        destination = obs['destination']

        # Node features: [is_current, is_destination]
        x = [[float(i == current_node), float(i == destination)] for i in range(env.num_nodes)]
        graph_data = graph_to_data(env.graph).to(device)
        graph_data.x = torch.tensor(x, dtype=torch.float).to(device)

        log_probs, values, rewards = [], [], []
        done = False
        total_reward = 0
        curr_step = 0

        while not done:
            logits, value_preds = model(graph_data)
            probs = logits[current_node]
            mask = mask_invalid_actions(env.graph, current_node, env.num_nodes).to(device)
            
            # Check for NaN in logits
            if torch.isnan(probs).any():
                print("Warning: NaN in logits")
                valid_neighbors = [nbr for nbr in env.graph.neighbors(current_node)
                                if not env.graph[current_node][nbr].get("regulatory_blocked", False)]
                action = random.choice(valid_neighbors) if valid_neighbors else random.randint(0, env.num_nodes - 1)
                log_prob = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                # Apply mask to logits
                masked_logits = probs * mask
                
                # Normalize logits
                logits_sum = masked_logits.sum()
                if logits_sum.item() == 0:
                    action = random.choice([i for i in range(env.num_nodes) if mask[i].item() > 0])
                    log_prob = torch.tensor(0.0, device=device, requires_grad=True)
                else:
                    probs = F.softmax(masked_logits, dim=-1)
                    
                    # Check if probabilities are valid
                    if torch.isnan(probs).any():
                        print("Warning: NaN in probabilities")
                        valid_neighbors = [nbr for nbr in env.graph.neighbors(current_node)
                                        if not env.graph[current_node][nbr].get("regulatory_blocked", False)]
                        action = random.choice(valid_neighbors) if valid_neighbors else random.randint(0, env.num_nodes - 1)
                        log_prob = torch.tensor(0.0, device=device, requires_grad=True)
                    else:
                        dist = torch.distributions.Categorical(probs)
                        action = dist.sample()
                        log_prob = dist.log_prob(action)

            obs, reward, term, trunc, _ = env.step(action.item())
            done = term or trunc
            current_node = obs['current_node']

            log_probs.append(log_prob)
            values.append(value_preds[current_node])
            rewards.append(reward)
            total_reward += reward
            curr_step += 1

            # Update graph features
            graph_data.x = torch.tensor([
                [float(i == current_node), float(i == destination)] for i in range(env.num_nodes)
            ], dtype=torch.float).to(device)

        episode_rewards.append(total_reward)

        # GAE and returns
        with torch.no_grad():
            _, next_values = model(graph_data)
            next_value = next_values[current_node]

        values = torch.stack(values + [next_value])
        returns, advantages = [], []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        advantages = torch.tensor(advantages, dtype=torch.float).to(device)
        returns = torch.tensor(returns, dtype=torch.float).to(device)
        log_probs = torch.stack(log_probs)
        values = values[:-1]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Losses
        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + 0.5 * value_loss

        # Zero gradients and clip them
        optimizer.zero_grad()
        
        # Clip gradients before backward pass
        torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        # Add gradient noise for stability
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * 1e-4
                param.grad.data.add_(noise)
        
        loss.backward()
        optimizer.step()

        # Logging
        if ep % 100 == 0:
            avg_r = np.mean(episode_rewards[-100:])
            print(f"[Episode {ep}] Avg Reward (100): {avg_r:.2f}, Last: {total_reward:.2f}, Steps: {curr_step}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "drift_actor_critic.pt")
    print("✅ Saved upgraded actor-critic model as drift_actor_critic.pt")

if __name__ == "__main__":
    train()
