import argparse
import networkx as nx
import numpy as np
import json
from pathlib import Path
import cugraph
import cudf
import warnings
import pickle

# Suppress warnings
warnings.filterwarnings("ignore")

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

    print(f"Loading graph from {input_file}...")
    G = nx.read_gml(str(input_file))
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Load author handles
    author_handles = {}
    if args.show_labels:
        try:
            print(f"Loading author handles...")
            with open(posts_file, 'r') as f:
                for line in f:
                    post = json.loads(line)
                    if 'author_did' in post and 'author_handle' in post:
                        author_handles[post['author_did']] = post['author_handle']
        except FileNotFoundError:
            print("Warning: Posts file not found. Skipping labels.")

    # Node interactions (Degrees)
    node_interactions = dict(G.degree(weight='weight'))

    # Filter nodes
    if args.min_interactions is not None:
        print(f"Filtering to nodes with ≥ {args.min_interactions} interactions...")
        nodes_to_keep = [n for n, d in node_interactions.items() if d >= args.min_interactions]
        G = G.subgraph(nodes_to_keep).copy()

    # Filter communities
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
    # Explicitly casting weights to float to avoid mixed types
    edge_data = []
    for u, v, d in G.edges(data=True):
        edge_data.append((node_to_int[u], node_to_int[v], float(d.get('weight', 1.0))))
    
    edge_df = cudf.DataFrame(edge_data, columns=['src', 'dst', 'weight'])
    
    cu_G = cugraph.Graph()
    cu_G.from_cudf_edgelist(edge_df, source='src', destination='dst', edge_attr='weight')

    print("Computing Force Atlas 2 layout on GPU...")
    pos_df = cugraph.force_atlas2(
        cu_G,
        max_iter=1000,
        outbound_attraction_distribution=True,
        lin_log_mode=False,
        prevent_overlapping=False,
        edge_weight_influence=1.0,
        jitter_tolerance=1.0,
        barnes_hut_optimize=True,
        barnes_hut_theta=1.0,
        scaling_ratio=2.0,
        gravity=10.0
    )

    # --- DATA SANITIZATION (CRITICAL STEP) ---
    print("Sanitizing data for CPU visualization...")
    
    # 1. Move to Pandas (CPU)
    pos_df_host = pos_df.to_pandas()
    
    # 2. Convert to dictionary of Native Python Floats
    # We avoid any numpy/pandas types in the final dictionary
    pos_map = {
        row.vertex: (float(row.x), float(row.y)) 
        for row in pos_df_host.itertuples(index=False)
    }

    # 3. Map back to original node IDs
    positions = {}
    for idx, node in int_to_node.items():
        if idx in pos_map:
            positions[node] = pos_map[idx]

    # Save layout and graph data to pickle for visualization
    print("Saving layout data...")
    layout_file = script_dir.parent / "data/bluesky/layout_data.pkl"
    layout_data = {
        'positions': positions,
        'graph': G,
        'node_interactions': node_interactions,
        'author_handles': author_handles,
        'show_labels': args.show_labels,
        'min_interactions': args.min_interactions,
        'min_community_size': args.min_community_size
    }
    with open(layout_file, 'wb') as f:
        pickle.dump(layout_data, f)

    print(f"Layout computed and saved to {layout_file}")
    print("Now running visualization in separate process...")

    # Clean up all CUDA objects
    del cu_G, edge_df, pos_df, pos_df_host, pos_map
    import gc
    gc.collect()

    # Run visualization in subprocess to avoid CUDA/matplotlib conflicts
    import subprocess
    viz_script = script_dir / "visualize_from_layout.py"
    result = subprocess.run(['python', str(viz_script), str(layout_file), str(output_file)],
                          capture_output=True, text=True)

    if result.returncode == 0:
        print("Success.")
    else:
        print("Visualization failed:")
        print(result.stderr)
        return

    # Clean up layout file
    layout_file.unlink()

if __name__ == "__main__":
    main()