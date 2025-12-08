import argparse
import networkx as nx

def main():
    parser = argparse.ArgumentParser(description='Find most influential nodes')
    parser.add_argument('--graph', type=str, default='../data/bluesky/user_interaction_graph_communities.gml')
    parser.add_argument('--top-n', type=int, default=10)
    args = parser.parse_args()

    print(f"Loading graph from {args.graph}...")
    G = nx.read_gml(args.graph)
    print(f"Loaded {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print("Computing Degree Centrality...")
    degree_centrality = nx.degree_centrality(G)

    sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:args.top_n]

    print(f"\n=== Top {args.top_n} Influential Nodes (Degree Centrality) ===")
    print(f"{'Rank':<5} {'Node':<15} {'Handle':<30} {'Score':<10}")
    print("-" * 65)
    
    for i, (node, score) in enumerate(sorted_nodes, 1):
        handle = G.nodes[node].get('author_handle', 'Unknown')
        print(f"{i:<5} {str(node):<15} {handle:<30} {score:.6f}")

if __name__ == "__main__":
    main()
