import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
wine_data = load_wine()
df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
correlation_matrix = df.corr()

# Create a network graph from the correlation matrix
G = nx.from_numpy_array(np.abs(correlation_matrix.values))

# Set labels
labels = {i: col for i, col in enumerate(correlation_matrix.columns)}

# Plotting the correlation graph
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)  # positions for all nodes

nx.draw_networkx_nodes(G, pos, node_size=1400,node_color='red')
nx.draw_networkx_edges(G, pos, width=correlation_matrix.values.flatten(), alpha=0.6)
nx.draw_networkx_labels(G, pos, labels, font_size=12)

plt.title('Correlation Network of wine quality Dataset')
plt.axis('off')
plt.show()
