
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
# 1. Create a directed graph
G = nx.DiGraph()

# 2. Add transaction nodes
G.add_nodes_from([1, 2, 3, 4, 5])

# 3. Add transaction connections (edges)
G.add_edges_from([
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 4),
    (4, 5)
])

# 4. Display nodes and edges
print("Nodes:", list(G.nodes()))
print("Edges:", list(G.edges()))

# 5. Calculate degree
print("\nDegree:", dict(G.degree()))

# 6. Calculate in-degree and out-degree
print("In-degree:", dict(G.in_degree()))
print("Out-degree:", dict(G.out_degree()))

# 7. Find neighbors of node 1
print("\nNeighbors of node 1:", list(G.neighbors(1)))

# 8. Create adjacency matrix
print("\nAdjacency Matrix:")
print(nx.to_numpy_array(G, nodelist=[1, 2, 3, 4, 5], dtype=int))

# 9. Visualize graph
nx.draw(
    G,
    with_labels=True,
    node_color="lightblue",
    node_size=1500,
    arrows=True
)
plt.title("Financial Transaction Graph")
plt.show()

# 10. Convert graph into PyTorch Geometric format


edge_index = torch.tensor(
    [[0, 0, 1, 2, 3],
     [1, 2, 3, 3, 4]],
    dtype=torch.long
)

x = torch.tensor(
    [[1.0], [2.0], [3.0], [4.0], [5.0]],
    dtype=torch.float
)

data = Data(x=x, edge_index=edge_index)

print("\nPyTorch Geometric Data:")
print(data)
print(data.edge_index)
