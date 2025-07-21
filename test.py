from torch_geometric.nn import SAGEConv
import torch
from torch_geometric.data import Data

conv = SAGEConv(8, 32)
data = Data(x=torch.randn(10, 8), edge_index=torch.randint(0, 10, (2, 20)))
print(conv(data).shape)
