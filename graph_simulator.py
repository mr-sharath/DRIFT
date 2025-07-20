# graph_simulator.py

import networkx as nx
import numpy as np
import random

class TransactionGraphSimulator:
    def __init__(self, num_nodes=10, edge_prob=0.3, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.num_nodes = num_nodes
        self.edge_prob = edge_prob
        self.graph = self._generate_graph()

    def _generate_graph(self):
        G = nx.erdos_renyi_graph(self.num_nodes, self.edge_prob, seed=42, directed=True)
        for u, v in G.edges():
            G[u][v]['fee'] = np.round(np.random.uniform(0.01, 0.10), 4)
            G[u][v]['capacity'] = np.random.randint(1000, 10000)
            G[u][v]['latency'] = np.round(np.random.uniform(1.0, 10.0), 2)
            G[u][v]['reliability'] = np.round(np.random.uniform(0.85, 1.0), 3)  # 85% to 100%
            G[u][v]['risk_score'] = np.round(np.random.uniform(0.0, 0.2), 3)   # 0 to 20% settlement risk
            G[u][v]['regulatory_blocked'] = random.random() < 0.05  # 5% chance blocked

        for n in G.nodes():
            G.nodes[n]['balance'] = np.random.randint(5000, 20000)
            G.nodes[n]['processing_power'] = np.random.randint(1, 10)
        return G

    def reset(self):
        self.graph = self._generate_graph()
        return self.graph

    def visualize_graph(self):
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', edge_color='gray')
        edge_labels = {
            (u, v): f"fee: {d['fee']}\ncap: {d['capacity']}\nrel: {d['reliability']}\nrisk: {d['risk_score']}"
            for u, v, d in self.graph.edges(data=True) if not d.get('regulatory_blocked', False)
        }
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        plt.title("Simulated Financial Transaction Network (Enhanced)")
        plt.show()

if __name__ == '__main__':
    sim = TransactionGraphSimulator(num_nodes=12, edge_prob=0.4)
    sim.visualize_graph()
