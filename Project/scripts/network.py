import argparse
import networkx as nx
import pandas as pd
from pathlib import Path
from collections import defaultdict
from preprocess_data import preprocess_posts_and_interactions
from metrics import calculate_edge_betweenness
from backboning import high_salience_skeleton


def create_uri_to_author_mapping(posts):
    post_to_author = {}
    for _, row in posts.iterrows():
        post_to_author[row['uri']] = row['author_did']
    return post_to_author


def build_interaction_graph(posts, interactions, silent=False, debug=False):
    # Create URI to author mapping
    post_to_author = create_uri_to_author_mapping(posts)

    if not silent:
        print(f"\nLoaded {len(post_to_author)} posts")

    # Build directed graph
    G = nx.DiGraph()

    # Track interaction statistics
    interaction_counts = defaultdict(lambda: defaultdict(int))
    interaction_types = defaultdict(int)
    missing_posts = 0

    # Process interactions
    for _, interaction in interactions.iterrows():
        src_user = interaction['src_did']
        dst_post = interaction['dst_uri']
        edge_type = interaction['edge']

        # Lookup the author of the target post
        if dst_post in post_to_author:
            dst_user = post_to_author[dst_post]

            # No self loops
            if src_user != dst_user:
                # Track edge type and count
                interaction_counts[(src_user, dst_user)][edge_type] += 1
                interaction_types[edge_type] += 1
        else:
            missing_posts += 1

    # Print interaction statistics
    if not silent:
        print(f"\n=== Interaction statistics ===")
        for edge_type, count in interaction_types.items():
            print(f"  {edge_type}: {count:,}")
        print(f"  Missing posts: {missing_posts:,}")

    if debug:
        print(f"\nDebug: Total unique user pairs: {len(interaction_counts)}")

    # Add edges to graph with weights and edge type information
    for (src, dst), edge_types in interaction_counts.items():
        total_weight = sum(edge_types.values())

        G.add_edge(src, dst,
                   weight=total_weight,
                   likes=edge_types.get('like', 0),
                   reposts=edge_types.get('repost', 0),
                   replies=edge_types.get('reply', 0))

    if not silent:
        print(f"\n=== Graph created ===")
        print(f"  Nodes (users): {G.number_of_nodes():,}")
        print(f"  Edges (user-to-user interactions): {G.number_of_edges():,}")
        print(f"  Density: {nx.density(G):.6f}")

    stats = {
        'interaction_types': interaction_types,
        'missing_posts': missing_posts,
        'interaction_counts': interaction_counts
    }

    return G, stats


def save_graph(G, output_path, silent=False):
    nx.write_gml(G, output_path)
    if not silent:
        print(f"\nGraph saved to: {output_path}")


def extract_backbone(G, quantile=0.75, silent=False, debug=False):
    if not silent:
        print("\n=== Extracting network backbone ===")

    # Convert to undirected
    G_undir = G.to_undirected()

    if not silent:
        print(f"Step 1: Calculating edge betweenness centrality for {G_undir.number_of_edges():,} edges...")

    # Calculate edge betweenness centrality
    edge_betweenness = calculate_edge_betweenness(G, weight='weight')

    # Normalize betweenness to range [1, 100]
    min_betweenness = min(edge_betweenness.values())
    max_betweenness = max(edge_betweenness.values())

    edge_weights_normalized = {
        edge: 1 + 99 * (betweenness - min_betweenness) / (max_betweenness - min_betweenness)
        for edge, betweenness in edge_betweenness.items()
    }

    if debug:
        print(f"\nDebug: Betweenness range: [{min_betweenness:.6f}, {max_betweenness:.6f}]")

    # Convert to edge table for backboning
    table = nx.to_pandas_edgelist(G_undir)

    # Add normalized betweenness weights
    table['weights_betweenness'] = table.apply(
        lambda row: edge_betweenness.get((row['source'], row['target']), 0), axis=1
    )
    table['weight_1'] = table.apply(
        lambda row: edge_weights_normalized.get((row['source'], row['target']), 0), axis=1
    )

    # Apply thresholding for keeping edges above quantile
    threshold = table['weight_1'].quantile(quantile)
    filtered_table = table[table['weight_1'] >= threshold]

    if not silent:
        print(f"Step 2: Filtered to {len(filtered_table):,} edges (top {(1-quantile)*100:.0f}%)")

    # Rename columns for backboning module
    table_renamed = filtered_table.rename(columns={"source": "src", "target": "trg", "weights_betweenness": "nij"})

    if not silent:
        print("Step 3: Running High Salience Skeleton algorithm...")

    # Run High Salience Skeleton
    hss_table = high_salience_skeleton(table_renamed, undirected=True)

    # Apply final thresholding on HSS scores
    threshold_hss = hss_table['score'].quantile(quantile)
    filtered_hss = hss_table[hss_table['score'] >= threshold_hss]

    if debug:
        print(f"\nDebug: HSS score threshold (quantile {quantile}): {threshold_hss:.6f}")
        print(f"Debug: Edges after HSS filtering: {len(filtered_hss):,}")

    # Build the backbone graph
    backbone = nx.from_pandas_edgelist(
        filtered_hss,
        source="src",
        target="trg",
        edge_attr=["nij", "score"],
        create_using=nx.Graph()
    )

    if not silent:
        print(f"\nBackbone extracted:")
        print(f"  Original edges: {G_undir.number_of_edges():,}")
        print(f"  Backbone edges: {backbone.number_of_edges():,} ({backbone.number_of_edges()/G_undir.number_of_edges()*100:.1f}%)")
        print(f"  Original nodes: {G_undir.number_of_nodes():,}")
        print(f"  Backbone nodes: {backbone.number_of_nodes():,}")

    return backbone


def analyze_graph(G, silent=False, debug=False):
    if not silent:
        print("\n=== Graph Statistics ===")

    # Degree distribution
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    if not silent:
        print(f"\nIn-degree (received interactions):")
        print(f"  Mean: {sum(in_degrees.values()) / len(in_degrees):.2f}")
        print(f"  Median: {sorted(in_degrees.values())[len(in_degrees) // 2]}")
        print(f"  Max: {max(in_degrees.values())}")
        print(f"  Min: {min(in_degrees.values())}")

        print(f"\nOut-degree (initiated interactions):")
        print(f"  Mean: {sum(out_degrees.values()) / len(out_degrees):.2f}")
        print(f"  Median: {sorted(out_degrees.values())[len(out_degrees) // 2]}")
        print(f"  Max: {max(out_degrees.values())}")
        print(f"  Min: {min(out_degrees.values())}")

    if debug:
        # Find most connected users
        print(f"\nTop 5 users by in-degree (most interactions received):")
        top_in = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        for user, degree in top_in:
            print(f"  {user}: {degree}")

        print(f"\nTop 5 users by out-degree (most interactions initiated):")
        top_out = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        for user, degree in top_out:
            print(f"  {user}: {degree}")

    return {
        'in_degrees': in_degrees,
        'out_degrees': out_degrees
    }


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Build user interaction network from Bluesky data')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with verbose output')
    parser.add_argument('--silent', action='store_true', help='Silent mode - minimal output')
    parser.add_argument('--backbone', action='store_true', help='Extract network backbone using edge betweenness')
    parser.add_argument('--backbone-quantile', type=float, default=0.75,
                        help='Quantile threshold for backbone extraction (default: 0.75)')
    args = parser.parse_args()

    # Define data paths
    data_dir = Path("../data/bluesky")
    posts_file = data_dir / "ca_fire_20250101_20250207.jsonl"
    interactions_file = data_dir / "interactions_20250101_20250207.jsonl"

    # Preprocess data
    if not args.silent:
        print("=== Preprocessing data ===")
    posts, interactions = preprocess_posts_and_interactions(str(posts_file), str(interactions_file))

    # Build interaction graph
    G, stats = build_interaction_graph(posts, interactions, silent=args.silent, debug=args.debug)

    # Save full graph
    graph_output = data_dir / "user_interaction_graph.gml"
    save_graph(G, graph_output, silent=args.silent)

    # Analyze full graph
    analysis = analyze_graph(G, silent=args.silent, debug=args.debug)

    # Extract backbone if requested
    if args.backbone:
        backbone = extract_backbone(G, quantile=args.backbone_quantile,
                                   silent=args.silent, debug=args.debug)

        # Save backbone
        backbone_output = data_dir / "user_interaction_graph_backbone.gml"
        save_graph(backbone, backbone_output, silent=args.silent)


if __name__ == "__main__":
    main()
