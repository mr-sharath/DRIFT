# evaluate_agent.py

import torch
import numpy as np
from drift_env import TransactionRoutingEnv
from train_drift_agent import GNNPolicy, graph_to_data
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt


def evaluate_policy(model, env, device, episodes=100):
    model.eval()
    stats = defaultdict(list)

    for ep in range(episodes):
        obs, _ = env.reset()
        graph_data = graph_to_data(env.graph).to(device)
        current_node = obs['current_node']
        destination = obs['destination']
        done = False
        ep_reward, steps = 0, 0

        while not done:
            logits = model(graph_data)
            probs = torch.softmax(logits, dim=0)
            action_probs = probs[current_node]
            action = torch.multinomial(action_probs, 1).item()

            obs, reward, term, trunc, _ = env.step(action)
            current_node = obs['current_node']
            ep_reward += reward
            steps += 1
            done = term or trunc

        stats['gnn_reward'].append(ep_reward)
        stats['gnn_success'].append(current_node == destination)
        stats['gnn_steps'].append(steps)

    return stats


def evaluate_random(env, episodes=100):
    stats = defaultdict(list)
    for ep in range(episodes):
        obs, _ = env.reset()
        current_node = obs['current_node']
        destination = obs['destination']
        done = False
        ep_reward, steps = 0, 0

        while not done:
            action = env.action_space.sample()
            obs, reward, term, trunc, _ = env.step(action)
            current_node = obs['current_node']
            ep_reward += reward
            steps += 1
            done = term or trunc

        stats['rand_reward'].append(ep_reward)
        stats['rand_success'].append(current_node == destination)
        stats['rand_steps'].append(steps)

    return stats


def evaluate_dijkstra(env, episodes=100):
    stats = defaultdict(list)
    for ep in range(episodes):
        obs, _ = env.reset()
        G = env.graph.copy()
        source, destination = obs['current_node'], obs['destination']
        for u, v in list(G.edges()):
            if G[u][v].get("regulatory_blocked", False):
                G.remove_edge(u, v)

        success = False
        try:
            path = nx.shortest_path(G, source=source, target=destination, weight='fee')
            total_fee = sum(G[path[i]][path[i+1]]['fee'] for i in range(len(path)-1))
            steps = len(path) - 1
            reward = 5.0 - total_fee
            success = True
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            reward = -5.0
            steps = 0

        stats['dijkstra_reward'].append(reward)
        stats['dijkstra_success'].append(success)
        stats['dijkstra_steps'].append(steps)

    return stats


def plot_metrics(gnn_stats, rand_stats, dijkstra_stats):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Avg Reward")
    plt.bar(["GNN", "Random", "Dijkstra"], [
        np.mean(gnn_stats['gnn_reward']),
        np.mean(rand_stats['rand_reward']),
        np.mean(dijkstra_stats['dijkstra_reward'])
    ])

    plt.subplot(1, 3, 2)
    plt.title("Success Rate")
    plt.bar(["GNN", "Random", "Dijkstra"], [
        np.mean(gnn_stats['gnn_success']),
        np.mean(rand_stats['rand_success']),
        np.mean(dijkstra_stats['dijkstra_success'])
    ])

    plt.subplot(1, 3, 3)
    plt.title("Avg Steps")
    plt.bar(["GNN", "Random", "Dijkstra"], [
        np.mean(gnn_stats['gnn_steps']),
        np.mean(rand_stats['rand_steps']),
        np.mean(dijkstra_stats['dijkstra_steps'])
    ])

    plt.tight_layout()
    plt.savefig("eval_comparison.png")
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TransactionRoutingEnv()

    model = GNNPolicy(in_channels=2, hidden_channels=32, out_channels=env.num_nodes)
    model.load_state_dict(torch.load("drift_policy.pt", map_location=device))
    model.to(device)

    print("Evaluating GNN policy...")
    gnn_stats = evaluate_policy(model, env, device, episodes=100)

    print("Evaluating random agent...")
    rand_stats = evaluate_random(env, episodes=100)

    print("Evaluating Dijkstra baseline...")
    dijkstra_stats = evaluate_dijkstra(env, episodes=100)

    plot_metrics(gnn_stats, rand_stats, dijkstra_stats)

if __name__ == '__main__':
    main()
