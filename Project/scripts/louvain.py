import argparse
import json
import networkx as nx
from collections import defaultdict
from networkx.algorithms.community import modularity
import community as community_louvain


def main():
    parser = argparse.ArgumentParser(description='Louvain community detection')
    parser.add_argument('--resolution', type=float, default=1.0, help='Resolution parameter (default: 1.0)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--graph', type=str, default='../data/bluesky/user_interaction_graph_undirected.gml',
                        help='Input graph file')
    parser.add_argument('--output', type=str, default='../data/bluesky/communities.json',
                        help='Output communities JSON file')
    parser.add_argument('--output-graph', type=str,
                        default='../data/bluesky/user_interaction_graph_communities.gml',
                        help='Output graph with community attributes')
    args = parser.parse_args()

    # Load graph
    G = nx.read_gml(args.graph)
    print(f"Loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # extract GCC
    connected_components = list(nx.connected_components(G))
    giant_component = max(connected_components, key=len)
    G_connected = G.subgraph(giant_component).copy()

    coverage = (G_connected.number_of_nodes() / G.number_of_nodes()) * 100
    print(f"Giant component: {G_connected.number_of_nodes():,} nodes, {G_connected.number_of_edges():,} edges ({coverage:.1f}% coverage)")

    # Run Louvain
    print(f"Running Louvain (resolution={args.resolution})...")
    partition_dict = community_louvain.best_partition(
        G_connected,
        resolution=args.resolution,
        random_state=args.seed
    )

    # Convert to communities list
    comm_dict = defaultdict(set)
    for node, comm_id in partition_dict.items():
        comm_dict[comm_id].add(node)
    communities = list(comm_dict.values())

    # Calculate modularity
    mod_score = modularity(G_connected, communities)

    print(f"Communities: {len(communities)}")
    print(f"Modularity: {mod_score:.4f}")

    # Display top 3 communities with their leaders
    community_sizes = sorted([(i, len(c)) for i, c in enumerate(communities)], key=lambda x: x[1], reverse=True)
    print(f"\nTop 3 communities:")
    for rank, (comm_idx, size) in enumerate(community_sizes[:3], 1):
        community = communities[comm_idx]
        pct = (size / G_connected.number_of_nodes()) * 100
        print(f"\n  {rank}. Community {comm_idx}: {size:,} nodes ({pct:.1f}%)")

        # Find top leader by degree
        leader, degree = max([(node, G_connected.degree(node)) for node in community],
                            key=lambda x: x[1])

        # Get author_handle if available
        handle = G_connected.nodes[leader].get('author_handle', 'Unknown')
        print(f"     Leader: {handle} ({degree} connections)")

    # Add community attributes to graph
    for node in G_connected.nodes():
        if node in partition_dict:
            G_connected.nodes[node]['community'] = partition_dict[node]

    # Save outputs
    communities_list = [list(c) for c in communities]
    output_data = {
        'communities': communities_list,
        'partition': partition_dict,
        'num_communities': len(communities),
        'modularity': mod_score
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved: {args.output}")

    nx.write_gml(G_connected, args.output_graph)
    print(f"Saved: {args.output_graph}")


if __name__ == "__main__":
    main()
