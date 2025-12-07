import argparse
import networkx as nx
import pandas as pd
from pathlib import Path
from collections import defaultdict
from preprocess_data import preprocess_posts_and_interactions


def create_uri_to_author_mapping(posts):
    post_to_author = {}
    for _, row in posts.iterrows():
        post_to_author[row['uri']] = row['author_did']
    return post_to_author


def create_did_to_handle_mapping(posts):
    """Create mapping from author_did to author_handle"""
    did_to_handle = {}
    for _, row in posts.iterrows():
        if 'author_did' in row and 'author_handle' in row:
            did_to_handle[row['author_did']] = row['author_handle']
    return did_to_handle


def build_interaction_graph(posts, interactions, silent=False, debug=False):
    # Create URI to author mapping
    post_to_author = create_uri_to_author_mapping(posts)

    # Create DID to handle mapping
    did_to_handle = create_did_to_handle_mapping(posts)

    if not silent:
        print(f"\nLoaded {len(post_to_author)} posts")
        print(f"Loaded {len(did_to_handle)} author handles")

    # Build directed graph
    G_directed = nx.DiGraph()

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

        G_directed.add_edge(src, dst,
                           weight=total_weight,
                           likes=edge_types.get('like', 0),
                           reposts=edge_types.get('repost', 0),
                           replies=edge_types.get('reply', 0))

    # Add node attributes to directed graph
    for node in G_directed.nodes():
        # Add in-degree and out-degree
        G_directed.nodes[node]['in_degree'] = G_directed.in_degree(node)
        G_directed.nodes[node]['out_degree'] = G_directed.out_degree(node)

        # Add author_handle if available
        if node in did_to_handle:
            G_directed.nodes[node]['author_handle'] = did_to_handle[node]

    # Create undirected version
    G_undirected = G_directed.to_undirected()

    # Add node attributes to undirected graph
    for node in G_undirected.nodes():
        # For undirected graph, we still keep the original in/out degrees from directed graph
        G_undirected.nodes[node]['in_degree'] = G_directed.in_degree(node)
        G_undirected.nodes[node]['out_degree'] = G_directed.out_degree(node)
        G_undirected.nodes[node]['degree'] = G_undirected.degree(node)

        # Add author_handle if available
        if node in did_to_handle:
            G_undirected.nodes[node]['author_handle'] = did_to_handle[node]

    if not silent:
        print(f"\n=== Directed graph created ===")
        print(f"  Nodes (users): {G_directed.number_of_nodes():,}")
        print(f"  Edges (user-to-user interactions): {G_directed.number_of_edges():,}")
        print(f"  Density: {nx.density(G_directed):.6f}")

        print(f"\n=== Undirected graph created ===")
        print(f"  Nodes (users): {G_undirected.number_of_nodes():,}")
        print(f"  Edges (user-to-user interactions): {G_undirected.number_of_edges():,}")
        print(f"  Density: {nx.density(G_undirected):.6f}")

    stats = {
        'interaction_types': interaction_types,
        'missing_posts': missing_posts,
        'interaction_counts': interaction_counts
    }

    return G_directed, G_undirected, stats


def save_graph(G, output_path, silent=False):
    nx.write_gml(G, output_path)
    if not silent:
        print(f"\nGraph saved to: {output_path}")


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
    args = parser.parse_args()

    # Define data paths
    data_dir = Path("../data/bluesky")
    posts_file = data_dir / "posts_all.jsonl"
    interactions_file = data_dir / "interactions_all.jsonl"

    # Preprocess data
    if not args.silent:
        print("=== Preprocessing data ===")
    posts, interactions = preprocess_posts_and_interactions(str(posts_file), str(interactions_file))

    # Build interaction graphs (both directed and undirected)
    G_directed, G_undirected, stats = build_interaction_graph(posts, interactions, silent=args.silent, debug=args.debug)

    # Save directed graph
    directed_graph_output = data_dir / "user_interaction_graph_directed.gml"
    save_graph(G_directed, directed_graph_output, silent=args.silent)

    # Save undirected graph
    undirected_graph_output = data_dir / "user_interaction_graph_undirected.gml"
    save_graph(G_undirected, undirected_graph_output, silent=args.silent)

    # Analyze directed graph
    analysis = analyze_graph(G_directed, silent=args.silent, debug=args.debug)


if __name__ == "__main__":
    main()
