import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from fa2_modified import ForceAtlas2


def main():
    print("Visualizing communities (clustered version)...")
    # Paths
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data/bluesky"
    input_file = data_dir / "user_interaction_graph_with_communities.gml"
    output_file = script_dir.parent / "figures/community_visualization_clustered.png"
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading graph from {input_file}...")
    G = nx.read_gml(str(input_file))
    
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Calculate node degrees (interactions)
    print("Calculating node interactions...")
    node_interactions = dict(G.degree(weight='weight'))
    
    # Filter nodes with < 10 interactions
    print("Filtering nodes with < 10 interactions...")
    nodes_to_keep = [n for n, interactions in node_interactions.items() if interactions >= 10]
    G_filtered = G.subgraph(nodes_to_keep).copy()
    
    print(f"Filtered graph has {G_filtered.number_of_nodes()} nodes and {G_filtered.number_of_edges()} edges.")
    
    # ========================================
    # KEEP ONLY TOP-K EDGES PER NODE
    # ========================================
    # This removes weak inter-community edges and helps communities cluster
    TOP_K = 3  # Keep only the top 3 strongest connections per node
    
    print(f"Keeping only top {TOP_K} edges per node...")
    
    edges_to_keep = set()
    for node in G_filtered.nodes():
        # Get all edges for this node with their weights
        node_edges = []
        for neighbor in G_filtered.neighbors(node):
            weight = G_filtered[node][neighbor].get('weight', 1)
            node_edges.append((node, neighbor, weight))
        
        # Sort by weight and keep top K
        node_edges.sort(key=lambda x: x[2], reverse=True)
        for edge in node_edges[:TOP_K]:
            # Store as frozenset so (a,b) == (b,a)
            edges_to_keep.add(frozenset([edge[0], edge[1]]))
    
    # Create new graph with only top-K edges
    G_sparse = nx.Graph()
    G_sparse.add_nodes_from(G_filtered.nodes(data=True))
    
    for u, v, data in G_filtered.edges(data=True):
        if frozenset([u, v]) in edges_to_keep:
            G_sparse.add_edge(u, v, **data)
    
    print(f"Sparse graph has {G_sparse.number_of_nodes()} nodes and {G_sparse.number_of_edges()} edges.")

    # Prepare ForceAtlas2
    forceatlas2 = ForceAtlas2(
        outboundAttractionDistribution=True, 
        barnesHutOptimize=True,
        gravity=0.3,        # Lower gravity = less pull to center
        scalingRatio=10.0   # Higher = more spread out
    )

    print("Computing ForceAtlas2 layout...")
    positions = forceatlas2.forceatlas2_networkx_layout(G_sparse, pos=None, iterations=2000)

    # Visualization
    print("Visualizing...")
    plt.figure(figsize=(15, 15))
    
    # Get attributes for plotting
    node_attrs = G_sparse.nodes
    nodes_iter = list(G_sparse.nodes())

    communities = [node_attrs[n].get('community', 0) for n in nodes_iter]
    sizes = [node_interactions.get(n, 1) for n in nodes_iter]
    
    # Normalize sizes for better visualization
    sizes_array = np.array(sizes)
    sizes_scaled = np.log1p(sizes_array) * 20

    # Create a colormap
    unique_communities = sorted(list(set(communities)))
    num_communities = len(unique_communities)
    cmap = plt.colormaps.get_cmap('tab20')
    
    # Map community to color index
    comm_to_idx = {c: i for i, c in enumerate(unique_communities)}
    node_colors = [comm_to_idx[c] for c in communities]
    
    # Draw edges first (transparent)
    nx.draw_networkx_edges(
        G_sparse, 
        positions, 
        alpha=0.1, 
        edge_color='gray',
        arrows=False
    )
    
    # Draw nodes
    scatter = nx.draw_networkx_nodes(
        G_sparse, 
        positions, 
        node_size=sizes_scaled, 
        node_color=node_colors, 
        cmap=cmap, 
        alpha=0.8,
        linewidths=0.5,
        edgecolors='white'
    )
    
    # Add legend
    if num_communities <= 20:
        legend_elements = []
        for i, comm in enumerate(unique_communities):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Comm {comm}',
                                            markerfacecolor=cmap(i/num_communities), markersize=10))
        plt.legend(handles=legend_elements, loc='upper right', title="Communities")

    plt.title(f"Community Visualization (Top-{TOP_K} edges, Nodes >= 10 Interactions)\n{G_sparse.number_of_nodes()} Nodes, {G_sparse.number_of_edges()} Edges", fontsize=16)
    plt.axis('off')
    
    print(f"Saving figure to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print("Done.")

if __name__ == "__main__":
    main()
