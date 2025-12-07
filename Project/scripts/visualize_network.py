import argparse
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter


def load_graph(graph_path):
    """Load graph from GML file"""
    return nx.read_gml(graph_path)


def plot_degree_distribution(degrees, title, output_path, log_scale=True):
    """Plot degree distribution"""
    degree_counts = Counter(degrees.values())

    # Sort by degree
    degrees_sorted = sorted(degree_counts.items())
    x = [d[0] for d in degrees_sorted]
    y = [d[1] for d in degrees_sorted]

    plt.figure(figsize=(10, 6))
    plt.bar(x, y, alpha=0.7, edgecolor='black')
    plt.xlabel('Degree', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    if log_scale:
        plt.yscale('log')
        plt.ylabel('Frequency (log scale)', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_top_nodes(G, degrees, top_n, title, output_path, degree_type='in'):
    """Plot top nodes by degree"""
    # Sort nodes by degree
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Get node labels (author_handle if available, else node id)
    labels = []
    degree_values = []

    for node, degree in sorted_nodes:
        if 'author_handle' in G.nodes[node]:
            label = G.nodes[node]['author_handle']
        else:
            label = node[:10] + '...' if len(node) > 10 else node
        labels.append(label)
        degree_values.append(degree)

    # Create bar plot
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(labels))

    plt.barh(y_pos, degree_values, alpha=0.7, edgecolor='black')
    plt.yticks(y_pos, labels, fontsize=10)
    plt.xlabel('Degree', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.gca().invert_yaxis()  # Highest at top

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def print_degree_statistics(degrees, degree_type):
    """Print statistics about degree distribution"""
    values = list(degrees.values())
    print(f"\n=== {degree_type} Statistics ===")
    print(f"  Mean: {np.mean(values):.2f}")
    print(f"  Median: {np.median(values):.2f}")
    print(f"  Std Dev: {np.std(values):.2f}")
    print(f"  Min: {min(values)}")
    print(f"  Max: {max(values)}")
    print(f"  Total nodes: {len(values)}")


def print_top_nodes(G, degrees, top_n, degree_type):
    """Print top nodes with their handles"""
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_n]

    print(f"\n=== Top {top_n} Nodes by {degree_type} ===")
    for i, (node, degree) in enumerate(sorted_nodes, 1):
        handle = G.nodes[node].get('author_handle', 'N/A')
        print(f"  {i}. {handle} (degree: {degree})")


def visualize_network(graph_path, output_dir, top_n=20, silent=False):
    """Main visualization function"""
    # Load graph
    if not silent:
        print(f"Loading graph from: {graph_path}")

    G = load_graph(graph_path)

    if not silent:
        print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get degree dictionaries
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    # Print statistics
    if not silent:
        print_degree_statistics(in_degrees, "In-Degree")
        print_degree_statistics(out_degrees, "Out-Degree")
        print_top_nodes(G, in_degrees, top_n, "In-Degree")
        print_top_nodes(G, out_degrees, top_n, "Out-Degree")

    # Plot in-degree distribution
    plot_degree_distribution(
        in_degrees,
        "In-Degree Distribution (Received Interactions)",
        output_dir / "in_degree_distribution.png"
    )

    # Plot out-degree distribution
    plot_degree_distribution(
        out_degrees,
        "Out-Degree Distribution (Initiated Interactions)",
        output_dir / "out_degree_distribution.png"
    )

    # Plot top nodes by in-degree
    plot_top_nodes(
        G, in_degrees, top_n,
        f"Top {top_n} Users by In-Degree (Most Interactions Received)",
        output_dir / f"top_{top_n}_in_degree.png",
        degree_type='in'
    )

    # Plot top nodes by out-degree
    plot_top_nodes(
        G, out_degrees, top_n,
        f"Top {top_n} Users by Out-Degree (Most Interactions Initiated)",
        output_dir / f"top_{top_n}_out_degree.png",
        degree_type='out'
    )

    if not silent:
        print(f"\n=== Visualization complete ===")
        print(f"All plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize network degree distributions')
    parser.add_argument('--graph', type=str,
                       default='../data/bluesky/user_interaction_graph_directed.gml',
                       help='Path to graph GML file')
    parser.add_argument('--output', type=str,
                       default='../figures',
                       help='Output directory for plots')
    parser.add_argument('--top-n', type=int, default=20,
                       help='Number of top nodes to display (default: 20)')
    parser.add_argument('--silent', action='store_true',
                       help='Silent mode - minimal output')

    args = parser.parse_args()

    visualize_network(args.graph, args.output, args.top_n, args.silent)


if __name__ == "__main__":
    main()
