# gnn_encoder.py

"""
Spatial Graph Neural Network (GNN) Encoder for Waynex.
Uses Graph Attention Networks (GAT) / GraphSAGE to map real street network graphs,
node delivery demands, time windows, and dynamic traffic attributes into high-dimensional representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

HAS_PYG = False
try:
    from torch_geometric.nn import GATConv, SAGEConv
    from torch_geometric.data import Data as GraphData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class WaynexGNNEncoder(nn.Module):
    def __init__(self, in_features: int = 6, hidden_dim: int = 64, embed_dim: int = 64):
        """
        in_features: [lat_norm, lng_norm, is_depot, demand_norm, tw_start_norm, tw_end_norm]
        hidden_dim: GNN hidden layer channels
        embed_dim: Output node embedding dimension
        """
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        if HAS_PYG:
            self.conv1 = GATConv(in_features, hidden_dim, heads=2, concat=False)
            self.conv2 = SAGEConv(hidden_dim, embed_dim)
        else:
            # Fallback PyTorch Linear Graph Encoder if torch_geometric not installed
            self.fc1 = nn.Linear(in_features, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, embed_dim)

        self.node_projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None) -> torch.Tensor:
        """
        x: [num_nodes, in_features] tensor of node features
        edge_index: [2, num_edges] graph adjacency indices
        Returns: [num_nodes, embed_dim] high-dimensional node embeddings
        """
        if HAS_PYG and edge_index is not None and edge_index.numel() > 0:
            h = F.relu(self.conv1(x, edge_index))
            h = F.relu(self.conv2(h, edge_index))
        else:
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))

        node_embeddings = self.node_projector(h)
        return node_embeddings


def prepare_graph_node_features(
    coords: list,
    deliveries: list,
    bounds: dict = None
) -> torch.Tensor:
    """
    Construct node feature matrix [N, 6] from location coords and delivery specs.
    Features:
    0: Normalized Latitude
    1: Normalized Longitude
    2: Is Depot (1.0 if depot, 0.0 otherwise)
    3: Normalized Demand Volume
    4: Normalized Time Window Start
    5: Normalized Time Window End
    """
    n = len(coords)
    features = []

    # Calculate min/max for normalization
    lats = [c["lat"] for c in coords]
    lngs = [c["lng"] for c in coords]
    min_lat, max_lat = min(lats), max(lats) + 1e-6
    min_lng, max_lng = min(lngs), max(lngs) + 1e-6

    for i in range(n):
        lat_norm = (coords[i]["lat"] - min_lat) / (max_lat - min_lat)
        lng_norm = (coords[i]["lng"] - min_lng) / (max_lng - min_lng)
        
        is_depot = 1.0 if i == 0 else 0.0
        
        if i == 0:
            demand_norm = 0.0
            tw_start_norm = 0.0
            tw_end_norm = 1.0
        else:
            deliv = deliveries[i - 1]
            demand_norm = min(deliv.get("demand", 100) / 1000.0, 1.0)
            tw = deliv.get("time_window", [8, 18])
            tw_start_norm = tw[0] / 24.0
            tw_end_norm = tw[1] / 24.0

        features.append([lat_norm, lng_norm, is_depot, demand_norm, tw_start_norm, tw_end_norm])

    return torch.tensor(features, dtype=torch.float32)


def create_fully_connected_edge_index(num_nodes: int) -> torch.Tensor:
    """Create bidirectional edge index tensor for PyG."""
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edges.append([i, j])
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()
