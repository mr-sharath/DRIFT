import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx

from drift_env import TransactionRoutingEnv
from train_drift_agent3 import GNNActorCritic

def mask_invalid_actions(graph, current_node, num_nodes):
    valid_mask = torch.zeros(num_nodes)
    for neighbor in graph.neighbors(current_node):
        if not graph[current_node][neighbor].get("regulatory_blocked", False):
            valid_mask[neighbor] = 1
    return valid_mask

def evaluate_actor_critic(model, env, device, episodes=100):
    model.eval()
    total_rewards, success_count, steps_list = [], 0, []

    for _ in range(episodes):
        obs, _ = env.reset()
        current_node = obs["current_node"]
        destination = obs["destination"]

        done, episode_reward, steps = False, 0, 0

        while not done:
            # Generate node features [is_current, is_destination]
            x = torch.tensor([
                [1.0 if i == current_node else 0.0, 1.0 if i == destination else 0.0]
                for i in range(env.num_nodes)
            ], dtype=torch.float).to(device)

            edge_index = []
            for u, v in env.graph.edges():
                edge_index.append([u, v])
                edge_index.append([v, u])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)

            data = type('GraphData', (object,), {})()
            data.x = x
            data.edge_index = edge_index

            with torch.no_grad():
                logits, _ = model(data)
                probs = F.softmax(logits, dim=-1)[current_node]
                mask = mask_invalid_actions(env.graph, current_node, env.num_nodes).to(device)
                masked_probs = probs * mask

                if masked_probs.sum().item() <= 1e-6:
                    neighbors = [nbr for nbr in env.graph.neighbors(current_node)
                                 if not env.graph[current_node][nbr].get("regulatory_blocked", False)]
                    action = random.choice(neighbors) if neighbors else random.randint(0, env.num_nodes - 1)
                else:
                    masked_probs /= masked_probs.sum()
                    action = torch.multinomial(masked_probs, 1).item()

            obs, reward, term, trunc, _ = env.step(action)
            current_node = obs["current_node"]
            episode_reward += reward
            steps += 1
            done = term or trunc

        total_rewards.append(episode_reward)
        steps_list.append(steps)
        if current_node == destination:
            success_count += 1

    return {
        "avg_reward": np.mean(total_rewards),
        "success_rate": success_count / episodes,
        "avg_steps": np.mean(steps_list)
    }

def evaluate_random(env, episodes=100):
    total_rewards, success_count, steps_list = [], 0, []

    for _ in range(episodes):
        obs, _ = env.reset()
        current_node, destination = obs['current_node'], obs['destination']
        done, episode_reward, steps = False, 0, 0

        while not done:
            neighbors = list(env.graph.neighbors(current_node))
            action = random.choice(neighbors) if neighbors else random.randint(0, env.num_nodes - 1)
            obs, reward, term, trunc, _ = env.step(action)
            current_node = obs['current_node']
            episode_reward += reward
            steps += 1
            done = term or trunc

        total_rewards.append(episode_reward)
        steps_list.append(steps)
        if current_node == destination:
            success_count += 1

    return {
        "avg_reward": np.mean(total_rewards),
        "success_rate": success_count / episodes,
        "avg_steps": np.mean(steps_list)
    }

def evaluate_dijkstra(env, episodes=100):
    total_rewards, success_count, steps_list = [], 0, []

    for _ in range(episodes):
        obs, _ = env.reset()
        source, destination = obs['current_node'], obs['destination']
        G = env.graph.copy()
        G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d.get("regulatory_blocked", False)])

        try:
            path = nx.shortest_path(G, source=source, target=destination, weight='fee')
            episode_reward = 0
            for i in range(len(path) - 1):
                obs, reward, term, trunc, _ = env.step(path[i + 1])
                episode_reward += reward
                if term or trunc:
                    break
            steps_list.append(len(path) - 1)
            total_rewards.append(episode_reward)
            if obs["current_node"] == destination:
                success_count += 1
        except nx.NetworkXNoPath:
            total_rewards.append(-10)
            steps_list.append(env.max_steps)

    return {
        "avg_reward": np.mean(total_rewards),
        "success_rate": success_count / episodes,
        "avg_steps": np.mean(steps_list)
    }

def plot_results(results):
    labels = list(results.keys())
    avg_rewards = [results[k]["avg_reward"] for k in labels]
    success_rates = [results[k]["success_rate"] for k in labels]
    avg_steps = [results[k]["avg_steps"] for k in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, avg_rewards, width, label="Avg Reward")
    ax.bar(x, success_rates, width, label="Success Rate")
    ax.bar(x + width, avg_steps, width, label="Avg Steps")

    ax.set_ylabel("Metrics")
    ax.set_title("Actor-Critic vs Dijkstra vs Random")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()
    plt.savefig("eval_actorcritic_vs_others.png")
    print("✅ Saved comparison plot as eval_actorcritic_vs_others.png")
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TransactionRoutingEnv()

    model = GNNActorCritic(in_channels=2, hidden_channels=64, out_channels=env.num_nodes).to(device)
    model.load_state_dict(torch.load("drift_actor_critic.pt", map_location=device))

    print("🎯 Evaluating Actor-Critic Agent...")
    actor_critic_stats = evaluate_actor_critic(model, env, device)

    print("🎲 Evaluating Random Agent...")
    random_stats = evaluate_random(env)

    print("🧭 Evaluating Dijkstra Agent...")
    dijkstra_stats = evaluate_dijkstra(env)

    results = {
        "Actor-Critic": actor_critic_stats,
        "Dijkstra Agent": dijkstra_stats,
        "Random Agent": random_stats
    }

    for name, stat in results.items():
        print(f"{name}:\n  Avg Reward: {stat['avg_reward']:.2f} | Success Rate: {stat['success_rate']:.2f} | Avg Steps: {stat['avg_steps']:.2f}\n")

    plot_results(results)

if __name__ == "__main__":
    main()
