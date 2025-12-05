import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from fa2_modified import ForceAtlas2


def main():
    print("Visualizing communities...")
    # Paths
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data/bluesky"
    input_file = data_dir / "user_interaction_graph_with_communities.gml"
    output_file = script_dir.parent / "figures/community_visualization.png"
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading graph from {input_file}...")
    G = nx.read_gml(str(input_file))
    
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Calculate node degrees (interactions)
    # Using degree as a proxy for total interactions (in + out)
    # If 'weight' is present on edges, we can use weighted degree
    print("Calculating node interactions...")
    node_interactions = dict(G.degree(weight='weight'))
    
    # Filter nodes with < 10 interactions
    print("Filtering nodes with < 10 interactions...")
    nodes_to_keep = [n for n, interactions in node_interactions.items() if interactions >= 10]
    G_filtered = G.subgraph(nodes_to_keep).copy()
    
    print(f"Filtered graph has {G_filtered.number_of_nodes()} nodes and {G_filtered.number_of_edges()} edges.")
    

    # Prepare ForceAtlas2
    forceatlas2 = ForceAtlas2(outboundAttractionDistribution=True, barnesHutOptimize=True)

    print("Computing ForceAtlas2 layout...")
    positions = forceatlas2.forceatlas2_networkx_layout(G_filtered, pos=None, iterations=2000)

    # Visualization
    print("Visualizing...")
    plt.figure(figsize=(15, 15))
    
    # Get attributes for plotting
    node_attrs = G_filtered.nodes
    nodes_iter = G_filtered.nodes()

    communities = [node_attrs[n].get('community', 0) for n in nodes_iter]
    sizes = [node_interactions[n] for n in nodes_iter]
    
    # Normalize sizes for better visualization
    sizes_array = np.array(sizes)
    sizes_scaled = np.log1p(sizes_array) * 20

    # Create a colormap
    unique_communities = sorted(list(set(communities)))
    num_communities = len(unique_communities)
    cmap = plt.cm.get_cmap('tab20', num_communities)
    
    # Draw edges first (transparent)
    nx.draw_networkx_edges(
        G_filtered, 
        positions, 
        alpha=0.1, 
        edge_color='gray',
        arrows=False
    )
    
    # Draw nodes
    scatter = nx.draw_networkx_nodes(
        G_filtered, 
        positions, 
        node_size=sizes_scaled, 
        node_color=communities, 
        cmap=cmap, 
        alpha=0.8,
        linewidths=0.5,
        edgecolors='white'
    )
    
    if num_communities <= 20:
        legend_elements = []
        for i, comm in enumerate(unique_communities):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Comm {comm}',
                                            markerfacecolor=cmap(i/num_communities), markersize=10))
        plt.legend(handles=legend_elements, loc='upper right', title="Communities")

    plt.title(f"Community Visualization (Nodes >= 10 Interactions)\n{G_filtered.number_of_nodes()} Nodes", fontsize=16)
    plt.axis('off')
    
    print(f"Saving figure to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print("Done.")

if __name__ == "__main__":
    main()
