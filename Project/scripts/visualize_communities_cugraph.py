#!/usr/bin/env python3
"""
Visualize communities in the network using cuGraph Force Atlas 2 (GPU-accelerated).
Uses SVG backend to avoid Agg/Pillow compatibility issues, with optional PNG conversion.
"""
import argparse
import networkx as nx
import numpy as np
import json
from pathlib import Path
import cugraph
import cudf
import warnings
import matplotlib
matplotlib.use('svg')  # Use SVG backend - Agg/PNG has compatibility issues with Pillow 11.x
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore")


def visualize_graph(G, positions, node_interactions, author_handles, show_labels, 
                    min_interactions, min_community_size, output_file):
    """Create and save the visualization."""
    print("Preparing visual attributes...")
    nodes_iter = list(G.nodes())

    # Communities - ensure native Python ints
    communities = [int(G.nodes[n].get('community', 0)) for n in nodes_iter]

    # Node sizes based on interactions
    sizes_raw = np.array([node_interactions.get(n, 1) for n in nodes_iter])
    if len(sizes_raw) > 0:
        norm = (sizes_raw - sizes_raw.min()) / (sizes_raw.max() - sizes_raw.min() + 1e-9)
        sizes_final = (norm ** 3 * 990 + 10)
    else:
        sizes_final = np.array([10] * len(nodes_iter))
    sizes_list = sizes_final.tolist()

    # Labels for top nodes in each community
    labels_to_draw = {}
    if show_labels:
        top_nodes = {}
        for n in nodes_iter:
            c = G.nodes[n].get('community', 0)
            score = node_interactions.get(n, 0)
            if c not in top_nodes or score > top_nodes[c][1]:
                top_nodes[c] = (n, score)

        for c, (n, _) in top_nodes.items():
            lbl = author_handles.get(n, n)
            if lbl.endswith('.bsky.social'):
                lbl = lbl[:-13]
            labels_to_draw[n] = lbl

    # Plotting
    print("Plotting...")
    plt.figure(figsize=(15, 15))

    unique_comms = len(set(communities))
    cmap = plt.colormaps.get_cmap('hsv').resampled(unique_comms)

    # Draw edges (only for edges where both nodes have positions)
    valid_edges = [(u, v) for u, v in G.edges() if u in positions and v in positions]
    nx.draw_networkx_edges(G, positions, edgelist=valid_edges, alpha=0.1, edge_color='gray', arrows=False)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, positions,
        node_size=sizes_list,
        node_color=communities,
        cmap=cmap,
        alpha=0.8,
        linewidths=0.5,
        edgecolors='white'
    )

    # Draw labels
    if show_labels and labels_to_draw:
        nx.draw_networkx_labels(
            G, positions, labels_to_draw,
            font_size=8,
            font_color='black',
            font_weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7)
        )

    # Title
    title = f"Community Visualization (cuGraph FA2)\n{G.number_of_nodes()} Nodes"
    if min_interactions:
        title += f" (≥{min_interactions} interactions)"
    if min_community_size:
        title += f"\n(communities with ≥{min_community_size} nodes)"

    plt.title(title, fontsize=16)
    plt.axis('off')

    # Save as SVG first (Agg/PNG backend has compatibility issues with Pillow)
    print(f"Saving to {output_file}...")
    svg_path = output_file.with_suffix('.svg')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    print(f"SVG saved to {svg_path}")

    # Convert SVG to PNG using cairosvg
    if str(output_file).endswith('.png'):
        try:
            import cairosvg
            cairosvg.svg2png(url=str(svg_path), write_to=str(output_file), scale=4)
            svg_path.unlink()  # Remove intermediate SVG file
            print(f"PNG saved to {output_file}")
        except ImportError:
            print("Note: cairosvg not installed. Keeping SVG output only.")
            print("Install with: pip install cairosvg")

    print("Visualization complete!")


def main():
    parser = argparse.ArgumentParser(description='Visualize communities in the network using cuGraph Force Atlas 2')
    parser.add_argument('--min-interactions', type=int, default=None,
                        help='Filter to nodes with at least N interactions')
    parser.add_argument('--min-community-size', type=int, default=None,
                        help='Show only communities with at least N nodes')
    parser.add_argument('--show-labels', action='store_true',
                        help='Show author handle for the biggest node in each community')
    args = parser.parse_args()

    print("Visualizing communities (GPU Accelerated)...")
    
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data/bluesky"
    input_file = data_dir / "user_interaction_graph_communities.gml"
    posts_file = data_dir / "posts_all.jsonl"
    output_file = script_dir.parent / "figures/community_visualization_cugraph.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load graph
    print(f"Loading graph from {input_file}...")
    G = nx.read_gml(str(input_file))
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Load author handles
    author_handles = {}
    if args.show_labels:
        try:
            print("Loading author handles...")
            with open(posts_file, 'r') as f:
                for line in f:
                    post = json.loads(line)
                    if 'author_did' in post and 'author_handle' in post:
                        author_handles[post['author_did']] = post['author_handle']
        except FileNotFoundError:
            print("Warning: Posts file not found. Skipping labels.")

    # Node interactions (Degrees)
    node_interactions = dict(G.degree(weight='weight'))

    # Filter nodes by interaction count
    if args.min_interactions is not None:
        print(f"Filtering to nodes with ≥ {args.min_interactions} interactions...")
        nodes_to_keep = [n for n, d in node_interactions.items() if d >= args.min_interactions]
        G = G.subgraph(nodes_to_keep).copy()

    # Filter communities by size
    if args.min_community_size is not None:
        print(f"Filtering to communities with ≥ {args.min_community_size} nodes...")
        comm_counts = {}
        for n in G.nodes():
            c = G.nodes[n].get('community', 0)
            comm_counts[c] = comm_counts.get(c, 0) + 1
        
        valid_comms = {c for c, count in comm_counts.items() if count >= args.min_community_size}
        print(f"Found {len(valid_comms)} communities.")
        
        nodes_to_keep = [n for n in G.nodes() if G.nodes[n].get('community', 0) in valid_comms]
        G = G.subgraph(nodes_to_keep).copy()
        print(f"Filtered graph: {G.number_of_nodes()} nodes.")

    # --- GPU LAYOUT COMPUTATION ---
    print("Converting graph to cuGraph format...")
    
    # Map string IDs to integers for cugraph
    node_list = list(G.nodes())
    node_to_int = {node: i for i, node in enumerate(node_list)}
    int_to_node = {i: node for node, i in node_to_int.items()}

    # Build Edge DataFrame
    edge_data = [(node_to_int[u], node_to_int[v], float(d.get('weight', 1.0))) 
                 for u, v, d in G.edges(data=True)]
    edge_df = cudf.DataFrame(edge_data, columns=['src', 'dst', 'weight'])
    
    cu_G = cugraph.Graph()
    cu_G.from_cudf_edgelist(edge_df, source='src', destination='dst', edge_attr='weight')

    print("Computing Force Atlas 2 layout on GPU...")
    #pos_df = cugraph.force_atlas2(
    #    cu_G,
    #    max_iter=1000,
    #    outbound_attraction_distribution=True,
    #    lin_log_mode=False,
    #    prevent_overlapping=False,
    #    edge_weight_influence=1.0,
    #    jitter_tolerance=1.0,
    #    barnes_hut_optimize=True,
    #    barnes_hut_theta=1.0,
    #    scaling_ratio=2.0,
    #    gravity=10.0
    #)

    pos_df = cugraph.layout.force_atlas2(
        cu_G,
        max_iter=5000,
        pos_list=None,
        outbound_attraction_distribution=True,
        lin_log_mode=True,
        edge_weight_influence=0.5,
        jitter_tolerance=1.0,
        barnes_hut_optimize=True,
        barnes_hut_theta=0.5,
        scaling_ratio=100.0,
        gravity=0.5
    )

    # Convert positions to CPU (native Python floats)
    print("Converting layout to CPU...")
    pos_df_host = pos_df.to_pandas()
    pos_map = {row.vertex: (float(row.x), float(row.y)) 
               for row in pos_df_host.itertuples(index=False)}

    # Map back to original node IDs
    positions = {int_to_node[idx]: pos_map[idx] for idx in int_to_node if idx in pos_map}

    # Clean up CUDA objects
    del cu_G, edge_df, pos_df, pos_df_host, pos_map
    import gc
    gc.collect()

    # Create visualization
    visualize_graph(G, positions, node_interactions, author_handles, 
                   args.show_labels, args.min_interactions, args.min_community_size, 
                   output_file)


if __name__ == "__main__":
    main()