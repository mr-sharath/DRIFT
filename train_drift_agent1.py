# train_drift_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
from drift_env import TransactionRoutingEnv
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx

class GNNPolicy(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNPolicy, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.actor = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        logits = self.actor(x)
        return logits


def graph_to_data(graph):
    for node in graph.nodes():
        features = [
            graph.nodes[node]['balance'] / 20000,
            graph.nodes[node]['processing_power'] / 10
        ]
        graph.nodes[node]['x'] = torch.tensor(features, dtype=torch.float32)
    return from_networkx(graph)


def train(env, model, optimizer, device, num_episodes=10000):
    model.train()
    rewards = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        graph_data = graph_to_data(env.graph).to(device)
        current_node = obs['current_node']
        destination = obs['destination']
        done = False
        ep_reward = 0
        step = 0

        while not done:
            logits = model(graph_data)
            probs = torch.softmax(logits, dim=0)
            action_probs = probs[current_node]
            action = torch.multinomial(action_probs, 1).item()

            obs, reward, term, trunc, _ = env.step(action)
            ep_reward += reward
            current_node = obs['current_node']
            done = term or trunc
            step += 1

        rewards.append(ep_reward)
        if episode % 25 == 0:
            print(f"Episode {episode} Reward: {ep_reward:.3f}")

    return model, rewards


def main():
    env = TransactionRoutingEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNNPolicy(in_channels=2, hidden_channels=32, out_channels=env.num_nodes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    trained_model, rewards = train(env, model, optimizer, device)
    torch.save(trained_model.state_dict(), "drift_policy.pt")
    print("Model saved as drift_policy.pt")

if __name__ == '__main__':
    main()
