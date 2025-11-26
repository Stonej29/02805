import argparse
import networkx as nx
from pathlib import Path
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities, modularity


def load_graph(graph_path):
    return nx.read_gml(graph_path)


def prepare_graph_for_community_detection(G, silent=False):
    if not silent:
        print("Preparing graph for community detection...")

    # Convert to undirected
    G_undir = G.to_undirected()

    # Filter to keep only connected nodes
    G_connected = G_undir.subgraph([n for n in G_undir.nodes() if G_undir.degree(n) > 0]).copy()

    if not silent:
        print(f"Connected graph:")
        print(f"  Nodes: {G_connected.number_of_nodes():,}")
        print(f"  Edges: {G_connected.number_of_edges():,}")

    return G_connected


def detect_communities(G, silent=False):
    if not silent:
        print("\nApplying community detection (greedy modularity optimization)...")

    # Apply greedy modularity algorithm
    communities_generator = greedy_modularity_communities(G, weight='weight')
    communities = [set(c) for c in communities_generator]

    # Create partition dict
    partition_dict = {}
    for comm_id, community in enumerate(communities):
        for node in community:
            partition_dict[node] = comm_id

    # Calculate modularity
    mod_score = modularity(G, communities)

    if not silent:
        print(f"Community detection completed!")
        print(f"  Number of communities: {len(communities)}")
        print(f"  Average community size: {sum(len(c) for c in communities) / len(communities):.2f}")
        print(f"\nModularity score: {mod_score:.4f}")
        print(f"\nInterpretation:")
        print(f"  > 0.3: Strong community structure")
        print(f"  0.2-0.3: Moderate community structure")
        print(f"  < 0.2: Weak community structure")

    return communities, partition_dict, mod_score


def analyze_community_sizes(G, communities, silent=False):
    community_sizes = [len(c) for c in communities]
    community_sizes_sorted = sorted(community_sizes, reverse=True)

    stats = {
        'largest': community_sizes_sorted[0],
        'smallest': community_sizes_sorted[-1],
        'median': np.median(community_sizes),
        'mean': np.mean(community_sizes),
        'sizes_sorted': community_sizes_sorted
    }

    if not silent:
        print("\n=== Community Analysis ===")
        print(f"\nCommunity size statistics:")
        print(f"  Largest community: {stats['largest']:,} users")
        print(f"  Smallest community: {stats['smallest']:,} users")
        print(f"  Median community size: {stats['median']:.0f} users")
        print(f"  Mean community size: {stats['mean']:.2f} users")

        print(f"\nTop 10 largest communities:")
        for i, size in enumerate(community_sizes_sorted[:10], 1):
            pct = (size / G.number_of_nodes()) * 100
            print(f"  Community {i}: {size:,} users ({pct:.2f}%)")

    return stats


def find_top_users_in_communities(G, communities, partition_dict, top_n=5, silent=False, debug=False):
    # Get top communities by size
    top_communities = sorted(communities, key=len, reverse=True)[:top_n]

    results = {}

    if not silent or debug:
        print("\n=== Top Users in Largest Communities ===")

    for i, community in enumerate(top_communities, 1):
        # Calculate degree centrality within community
        community_degrees = [(node, G.degree(node)) for node in community]
        top_users = sorted(community_degrees, key=lambda x: x[1], reverse=True)[:5]

        results[i] = top_users

        if not silent or debug:
            print(f"\nCommunity {i} ({len(community):,} users):")
            for rank, (user, degree) in enumerate(top_users, 1):
                print(f"  {rank}. {user}: {degree} connections")

    return results


def save_communities(communities, partition_dict, output_path, silent=False):
    import json

    # Convert sets to lists for JSON serialization
    communities_list = [list(c) for c in communities]

    output_data = {
        'communities': communities_list,
        'partition': partition_dict,
        'num_communities': len(communities)
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    if not silent:
        print(f"\nCommunities saved to: {output_path}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Perform community detection on Bluesky interaction network')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with verbose output')
    parser.add_argument('--silent', action='store_true', help='Silent mode - minimal output')
    parser.add_argument('--graph', type=str, default='../data/bluesky/user_interaction_graph.gml',
                        help='Path to input graph file (default: ../data/bluesky/user_interaction_graph.gml)')
    parser.add_argument('--output', type=str, default='../data/bluesky/communities.json',
                        help='Path to output communities file (default: ../data/bluesky/communities.json)')
    args = parser.parse_args()

    # Define paths
    graph_path = Path(args.graph)
    output_path = Path(args.output)

    # Step 1: Load graph
    if not args.silent:
        print("=== Loading graph ===")
    G = load_graph(graph_path)
    if not args.silent:
        print(f"Loaded graph with {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges")

    # Step 2: Prepare graph for community detection
    G_connected = prepare_graph_for_community_detection(G, silent=args.silent)

    # Step 3: Detect communities
    communities, partition_dict, mod_score = detect_communities(G_connected, silent=args.silent)

    # Step 4: Analyze community sizes
    stats = analyze_community_sizes(G_connected, communities, silent=args.silent)

    # Step 5: Find top users in communities
    top_users = find_top_users_in_communities(G_connected, communities, partition_dict,
                                              top_n=5, silent=args.silent, debug=args.debug)

    # Step 6: Save communities
    save_communities(communities, partition_dict, output_path, silent=args.silent)


if __name__ == "__main__":
    main()
