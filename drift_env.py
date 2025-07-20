# drift_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from graph_simulator import TransactionGraphSimulator
import networkx as nx

class TransactionRoutingEnv(gym.Env):
    def __init__(self, num_nodes=12, max_steps=10):
        super(TransactionRoutingEnv, self).__init__()
        self.simulator = TransactionGraphSimulator(num_nodes=num_nodes)
        self.graph = self.simulator.graph
        self.num_nodes = num_nodes
        self.max_steps = max_steps

        self.source = None
        self.destination = None
        self.current_node = None
        self.step_count = 0

        self.action_space = spaces.Discrete(self.num_nodes)  # Pick next node to hop to
        self.observation_space = spaces.Dict({
            "current_node": spaces.Discrete(self.num_nodes),
            "destination": spaces.Discrete(self.num_nodes),
            "graph_matrix": spaces.Box(low=0, high=1, shape=(self.num_nodes, self.num_nodes), dtype=np.float32)
        })

    def reset(self, seed=None, options=None):
        self.graph = self.simulator.reset()
        self.source, self.destination = np.random.choice(self.num_nodes, size=2, replace=False)
        self.current_node = self.source
        self.step_count = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.step_count += 1
        reward = -0.1  # Small penalty to discourage looping
        terminated = False
        truncated = False

        if not self.graph.has_edge(self.current_node, action):
            reward = -1.0  # Invalid move
        elif self.graph[self.current_node][action].get("regulatory_blocked", False):
            reward = -1.5  # Illegal move
        else:
            edge_data = self.graph[self.current_node][action]
            fee = edge_data['fee']
            reliability = edge_data['reliability']
            risk = edge_data['risk_score']

            reward = (0.5 * reliability) - (0.5 * risk) - fee
            self.current_node = action

        if self.current_node == self.destination:
            reward += 5.0  # Big reward for success
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        adj = nx.to_numpy_array(self.graph, weight='capacity')
        norm_adj = adj / (adj.max() + 1e-6)  # Normalize
        return {
            "current_node": self.current_node,
            "destination": self.destination,
            "graph_matrix": norm_adj.astype(np.float32)
        }

if __name__ == "__main__":
    env = TransactionRoutingEnv()
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, term, trunc, _ = env.step(action)
        total_reward += reward
        done = term or trunc

    print("Episode completed with reward:", total_reward)
