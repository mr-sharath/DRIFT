# evaluate_agent.py (Final Robust Version)

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from drift_env import TransactionRoutingEnv
from train_drift_agent2 import GNNPolicy, graph_to_data, mask_invalid_actions
import random
import networkx as nx

def evaluate_policy(model, env, device, episodes=100):
    model.eval()
    total_rewards = []
    success_count = 0
    step_counts = []

    for _ in range(episodes):
        obs, _ = env.reset()
        graph_data = graph_to_data(env.graph).to(device)
        current_node = obs['current_node']
        destination = obs['destination']

        episode_reward = 0
        steps = 0
        done = False

        while not done:
            with torch.no_grad():
                logits = model(graph_data)
                probs = F.softmax(logits, dim=-1)[current_node]
                mask = mask_invalid_actions(env.graph, current_node, env.num_nodes).to(device)
                masked_probs = probs * mask

                if masked_probs.sum().item() <= 1e-6:
                    # fallback strategy if no valid actions
                    valid_neighbors = [nbr for nbr in env.graph.neighbors(current_node)
                                       if not env.graph[current_node][nbr].get("regulatory_blocked", False)]
                    action = random.choice(valid_neighbors) if valid_neighbors else random.randint(0, env.num_nodes - 1)
                else:
                    masked_probs /= masked_probs.sum()
                    action = torch.multinomial(masked_probs, 1).item()

            obs, reward, term, trunc, _ = env.step(action)
            current_node = obs['current_node']
            episode_reward += reward
            steps += 1
            done = term or trunc

        total_rewards.append(episode_reward)
        step_counts.append(steps)
        if obs['current_node'] == obs['destination']:
            success_count += 1

    return {
        "avg_reward": np.mean(total_rewards),
        "success_rate": success_count / episodes,
        "avg_steps": np.mean(step_counts)
    }

def evaluate_random(env, episodes=100):
    total_rewards = []
    success_count = 0
    step_counts = []

    for _ in range(episodes):
        obs, _ = env.reset()
        current_node = obs['current_node']
        destination = obs['destination']

        episode_reward = 0
        steps = 0
        done = False

        while not done:
            neighbors = list(env.graph.neighbors(current_node))
            if not neighbors:
                action = random.randint(0, env.num_nodes - 1)
            else:
                action = random.choice(neighbors)
            obs, reward, term, trunc, _ = env.step(action)
            current_node = obs['current_node']
            episode_reward += reward
            steps += 1
            done = term or trunc

        total_rewards.append(episode_reward)
        step_counts.append(steps)
        if obs['current_node'] == obs['destination']:
            success_count += 1

    return {
        "avg_reward": np.mean(total_rewards),
        "success_rate": success_count / episodes,
        "avg_steps": np.mean(step_counts)
    }

def evaluate_dijkstra(env, episodes=100):
    total_rewards = []
    success_count = 0
    step_counts = []

    for _ in range(episodes):
        obs, _ = env.reset()
        source = obs['current_node']
        destination = obs['destination']
        G = env.graph.copy()

        # remove regulatory-blocked edges
        edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d.get("regulatory_blocked", False)]
        G.remove_edges_from(edges_to_remove)

        try:
            path = nx.shortest_path(G, source=source, target=destination, weight='fee')
            episode_reward = 0
            for i in range(len(path) - 1):
                obs, reward, term, trunc, _ = env.step(path[i+1])
                episode_reward += reward
                if term or trunc:
                    break
            total_rewards.append(episode_reward)
            step_counts.append(len(path) - 1)
            if obs['current_node'] == obs['destination']:
                success_count += 1
        except nx.NetworkXNoPath:
            total_rewards.append(-10)
            step_counts.append(env.max_steps)

    return {
        "avg_reward": np.mean(total_rewards),
        "success_rate": success_count / episodes,
        "avg_steps": np.mean(step_counts)
    }

def plot_metrics(results):
    labels = list(results.keys())
    avg_rewards = [results[k]['avg_reward'] for k in labels]
    success_rates = [results[k]['success_rate'] for k in labels]
    avg_steps = [results[k]['avg_steps'] for k in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, avg_rewards, width, label='Avg Reward')
    ax.bar(x, success_rates, width, label='Success Rate')
    ax.bar(x + width, avg_steps, width, label='Avg Steps')

    ax.set_ylabel('Scores')
    ax.set_title('Evaluation Metrics by Agent')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.savefig("eval_comparison3.png")
    print("✅ Saved evaluation plot as eval_comparison3.png")
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TransactionRoutingEnv()

    print("Evaluating GNN policy...")
    model = GNNPolicy(in_channels=1, hidden_channels=32, out_channels=env.num_nodes).to(device)
    model.load_state_dict(torch.load("drift_policy.pt", map_location=device))
    gnn_stats = evaluate_policy(model, env, device, episodes=100)

    print("Evaluating random policy...")
    random_stats = evaluate_random(env, episodes=100)

    print("Evaluating dijkstra policy...")
    dijkstra_stats = evaluate_dijkstra(env, episodes=100)

    results = {
        "GNN Agent": gnn_stats,
        "Random Agent": random_stats,
        "Dijkstra Agent": dijkstra_stats
    }

    for agent, stats in results.items():
        print(f"{agent}:\n  Avg Reward: {stats['avg_reward']:.2f}\n  Success Rate: {stats['success_rate']:.2f}\n  Avg Steps: {stats['avg_steps']:.2f}\n")

    plot_metrics(results)

if __name__ == '__main__':
    main()