import argparse
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from fa2_modified import ForceAtlas2


def main():
    parser = argparse.ArgumentParser(description='Visualize communities in the network')
    parser.add_argument('--min-interactions', type=int, default=None,
                        help='Filter to nodes with at least N interactions (default: all nodes)')
    parser.add_argument('--min-community-size', type=int, default=None,
                        help='Show only communities with at least N nodes (default: all communities)')
    parser.add_argument('--show-labels', action='store_true',
                        help='Show author handle for the biggest node in each community')
    args = parser.parse_args()

    print("Visualizing communities...")
    # Paths
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data/bluesky"
    input_file = data_dir / "user_interaction_graph_communities.gml"
    posts_file = data_dir / "posts_all.jsonl"
    output_file = script_dir.parent / "figures/community_visualization.png"

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading graph from {input_file}...")
    G = nx.read_gml(str(input_file))

    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Load author handles if labels are requested
    author_handles = {}
    if args.show_labels:
        print(f"Loading author handles from {posts_file}...")
        with open(posts_file, 'r') as f:
            for line in f:
                post = json.loads(line)
                author_did = post.get('author_did')
                author_handle = post.get('author_handle')
                if author_did and author_handle:
                    author_handles[author_did] = author_handle
        print(f"Loaded {len(author_handles)} unique author handles from posts.")

    # Calculate node degrees (interactions)
    print("Calculating node interactions...")
    node_interactions = dict(G.degree(weight='weight'))

    # Filter nodes with minimum interactions if specified
    if args.min_interactions is not None:
        print(f"Filtering to nodes with at least {args.min_interactions} interactions...")
        nodes_to_keep = [node for node, interactions in node_interactions.items()
                        if interactions >= args.min_interactions]
        G = G.subgraph(nodes_to_keep).copy()
        print(f"Filtered graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Filter to communities with minimum size if specified
    if args.min_community_size is not None:
        print(f"Filtering to communities with at least {args.min_community_size} nodes...")
        # Count nodes in each community
        community_sizes = {}
        for node in G.nodes():
            comm = G.nodes[node].get('community', 0)
            community_sizes[comm] = community_sizes.get(comm, 0) + 1

        # Get communities that meet the size threshold
        valid_communities = [comm_id for comm_id, size in community_sizes.items() if size >= args.min_community_size]

        print(f"Found {len(valid_communities)} communities with ≥{args.min_community_size} nodes")
        print(f"Community sizes: {sorted([(comm_id, size) for comm_id, size in community_sizes.items() if comm_id in valid_communities], key=lambda x: x[1], reverse=True)}")

        # Filter to nodes in valid communities
        nodes_to_keep = [node for node in G.nodes() if G.nodes[node].get('community', 0) in valid_communities]
        G = G.subgraph(nodes_to_keep).copy()
        print(f"Filtered graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Prepare ForceAtlas2
    forceatlas2 = ForceAtlas2(outboundAttractionDistribution=True, barnesHutOptimize=True, gravity=10.0)

    print("Computing ForceAtlas2 layout...")
    positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=2000)

    # Visualization
    print("Visualizing...")
    plt.figure(figsize=(15, 15))

    # Get attributes for plotting
    node_attrs = G.nodes
    nodes_iter = G.nodes()

    communities = [node_attrs[n].get('community', 0) for n in nodes_iter]
    sizes = [node_interactions[n] for n in nodes_iter]

    # Normalize sizes for better visualization using exponential scaling
    # This makes the biggest nodes actually big and the rest very small
    sizes_array = np.array(sizes)

    # Normalize to 0-1 range
    sizes_normalized = (sizes_array - sizes_array.min()) / (sizes_array.max() - sizes_array.min())

    # Apply power to amplify differences (higher exponent = more extreme)
    sizes_powered = sizes_normalized ** 3

    # Scale to visualization range (min size 10, max size 1000)
    sizes_scaled = sizes_powered * 990 + 10

    # Create a colormap for many communities
    unique_communities = sorted(list(set(communities)))
    num_communities = len(unique_communities)

    # Use hsv colormap for many distinct colors (works well for 50+ communities)
    cmap = plt.cm.get_cmap('hsv', num_communities)

    # Find biggest node in each community for labeling
    labels_to_draw = {}
    if args.show_labels:
        print("Finding biggest node in each community...")
        community_top_nodes = {}
        for node in nodes_iter:
            comm = node_attrs[node].get('community', 0)
            interactions = node_interactions[node]
            if comm not in community_top_nodes or interactions > community_top_nodes[comm][1]:
                community_top_nodes[comm] = (node, interactions)

        # Create labels dictionary
        for comm, (node, interactions) in community_top_nodes.items():
            handle = author_handles.get(node, node)  # Use DID if no handle found
            # Simplify handle (remove .bsky.social if present)
            if handle.endswith('.bsky.social'):
                handle = handle[:-13]
            labels_to_draw[node] = handle
        print(f"Prepared {len(labels_to_draw)} labels for community leaders.")

    # Draw edges first (transparent)
    nx.draw_networkx_edges(
        G,
        positions,
        alpha=0.1,
        edge_color='gray',
        arrows=False
    )

    # Draw nodes
    scatter = nx.draw_networkx_nodes(
        G,
        positions,
        node_size=sizes_scaled,
        node_color=communities,
        cmap=cmap,
        alpha=0.8,
        linewidths=0.5,
        edgecolors='white'
    )

    # Draw labels if requested
    if args.show_labels and labels_to_draw:
        nx.draw_networkx_labels(
            G,
            positions,
            labels_to_draw,
            font_size=8,
            font_color='black',
            font_weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7)
        )

    title = f"Community Visualization\n{G.number_of_nodes()} Nodes"
    if args.min_interactions is not None:
        title += f" (≥{args.min_interactions} interactions)"
    if args.min_community_size is not None:
        title += f"\n(communities with ≥{args.min_community_size} nodes)"
    plt.title(title, fontsize=16)
    plt.axis('off')
    
    print(f"Saving figure to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print("Done.")

if __name__ == "__main__":
    main()
