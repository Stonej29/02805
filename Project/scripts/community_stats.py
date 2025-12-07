import argparse
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from collections import Counter


def load_communities_data(communities_file, graph_file):
    """Load community data and graph"""
    # Load community JSON
    with open(communities_file, 'r') as f:
        data = json.load(f)

    # Load graph
    G = nx.read_gml(graph_file)

    return data, G


def calculate_community_stats(communities, G):
    """Calculate various community statistics"""
    community_sizes = [len(comm) for comm in communities]

    stats = {
        'num_communities': len(communities),
        'total_nodes': sum(community_sizes),
        'mean_size': np.mean(community_sizes),
        'median_size': np.median(community_sizes),
        'std_size': np.std(community_sizes),
        'min_size': min(community_sizes),
        'max_size': max(community_sizes),
        'community_sizes': community_sizes
    }

    return stats


def print_basic_stats(stats, modularity):
    """Print basic community statistics"""
    print("\n" + "="*60)
    print("COMMUNITY STATISTICS")
    print("="*60)
    print(f"\nTotal Communities: {stats['num_communities']}")
    print(f"Total Nodes: {stats['total_nodes']:,}")
    print(f"Modularity Score: {modularity:.4f}")
    print(f"\nCommunity Size Statistics:")
    print(f"  Mean:   {stats['mean_size']:.2f}")
    print(f"  Median: {stats['median_size']:.0f}")
    print(f"  Std Dev: {stats['std_size']:.2f}")
    print(f"  Min:    {stats['min_size']}")
    print(f"  Max:    {stats['max_size']:,}")


def print_top_communities(communities, G, top_n=10):
    """Print top N communities by size with their leaders"""
    # Sort communities by size
    sorted_communities = sorted(enumerate(communities), key=lambda x: len(x[1]), reverse=True)

    total_nodes = sum(len(comm) for comm in communities)

    print(f"\n" + "="*60)
    print(f"TOP {min(top_n, len(communities))} LARGEST COMMUNITIES")
    print("="*60)

    for rank, (comm_idx, community) in enumerate(sorted_communities[:top_n], 1):
        size = len(community)
        pct = (size / total_nodes) * 100

        # Find leader (highest degree node)
        leader, degree = max([(node, G.degree(node)) for node in community],
                            key=lambda x: x[1])

        # Get handle
        handle = G.nodes[leader].get('author_handle', 'Unknown')

        print(f"\n{rank}. Community {comm_idx}")
        print(f"   Size: {size:,} nodes ({pct:.2f}% of network)")
        print(f"   Leader: {handle}")
        print(f"   Leader degree: {degree:,} connections")


def print_size_distribution(stats):
    """Print community size distribution"""
    sizes = stats['community_sizes']

    # Create bins
    bins = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, float('inf')]
    bin_labels = ['1', '2-4', '5-9', '10-19', '20-49', '50-99', '100-199', '200-499', '500-999', '1000+']

    # Count communities in each bin
    counts = [0] * len(bin_labels)
    for size in sizes:
        for i, (lower, upper) in enumerate(zip(bins[:-1], bins[1:])):
            if lower <= size < upper:
                counts[i] += 1
                break

    print(f"\n" + "="*60)
    print("COMMUNITY SIZE DISTRIBUTION")
    print("="*60)
    print(f"\n{'Size Range':<15} {'Count':<10} {'Percentage':<12} {'Bar'}")
    print("-" * 60)

    total_communities = len(sizes)
    for label, count in zip(bin_labels, counts):
        if count > 0:
            pct = (count / total_communities) * 100
            bar = '█' * int(pct / 2)  # Scale bar to fit
            print(f"{label:<15} {count:<10} {pct:>5.1f}%        {bar}")


def plot_size_distribution(stats, output_path):
    """Plot community size distribution"""
    sizes = sorted(stats['community_sizes'], reverse=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Rank-size plot (log-log)
    ax1.loglog(range(1, len(sizes) + 1), sizes, 'b-', linewidth=2, alpha=0.7)
    ax1.scatter(range(1, len(sizes) + 1), sizes, s=20, alpha=0.5, c='blue')
    ax1.set_xlabel('Community Rank', fontsize=12)
    ax1.set_ylabel('Community Size (nodes)', fontsize=12)
    ax1.set_title('Community Size Rank Distribution (log-log)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Histogram
    # Filter out very large communities for better visualization
    sizes_filtered = [s for s in sizes if s <= np.percentile(sizes, 95)]

    ax2.hist(sizes_filtered, bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Community Size (nodes)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Community Size Distribution (bottom 95%)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {output_path}")


def plot_top_communities(communities, G, output_path, top_n=15):
    """Plot bar chart of top N communities"""
    # Sort communities by size
    sorted_communities = sorted(enumerate(communities), key=lambda x: len(x[1]), reverse=True)

    # Get top N
    top_communities = sorted_communities[:top_n]

    # Prepare data
    labels = []
    sizes = []
    total_nodes = sum(len(comm) for comm in communities)

    for comm_idx, community in top_communities:
        size = len(community)

        # Find leader
        leader, _ = max([(node, G.degree(node)) for node in community],
                       key=lambda x: x[1])
        handle = G.nodes[leader].get('author_handle', 'Unknown')

        # Truncate long handles
        if len(handle) > 25:
            handle = handle[:22] + '...'

        label = f"C{comm_idx}: {handle}"
        labels.append(label)
        sizes.append(size)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(labels))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(labels)))

    bars = ax.barh(y_pos, sizes, color=colors, alpha=0.8, edgecolor='black')

    # Add percentage labels on bars
    for i, (bar, size) in enumerate(zip(bars, sizes)):
        pct = (size / total_nodes) * 100
        ax.text(bar.get_width() + max(sizes)*0.01, bar.get_y() + bar.get_height()/2,
                f'{size:,} ({pct:.1f}%)',
                va='center', fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_title(f'Top {top_n} Largest Communities by Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze community statistics')
    parser.add_argument('--communities', type=str,
                       default='../data/bluesky/communities.json',
                       help='Path to communities JSON file')
    parser.add_argument('--graph', type=str,
                       default='../data/bluesky/user_interaction_graph_communities.gml',
                       help='Path to graph with community attributes')
    parser.add_argument('--output', type=str,
                       default='../figures',
                       help='Output directory for plots')
    parser.add_argument('--top-n', type=int, default=15,
                       help='Number of top communities to display (default: 15)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')

    args = parser.parse_args()

    # Load data
    print("Loading community data...")
    data, G = load_communities_data(args.communities, args.graph)

    communities = [set(comm) for comm in data['communities']]
    modularity = data['modularity']

    # Calculate statistics
    print("Calculating statistics...")
    stats = calculate_community_stats(communities, G)

    # Print statistics
    print_basic_stats(stats, modularity)
    print_top_communities(communities, G, args.top_n)
    print_size_distribution(stats)

    # Generate plots
    if not args.no_plots:
        print("\nGenerating plots...")
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Size distribution plot
        plot_size_distribution(stats, output_dir / "community_size_distribution.png")

        # Top communities bar chart
        plot_top_communities(communities, G, output_dir / "top_communities.png", args.top_n)

        print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
