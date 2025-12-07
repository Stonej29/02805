#!/usr/bin/env python3
"""
Helper script to visualize graph from pre-computed layout.
This runs in a separate process to avoid CUDA/matplotlib conflicts.
Uses SVG backend to avoid Agg/Pillow compatibility issues.
"""
import sys
import pickle
import matplotlib
matplotlib.use('svg')  # Use SVG backend - Agg/PNG has compatibility issues
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path
import tempfile
import os


def sanitize_value(val):
    """Convert numpy/pandas types to native Python types."""
    if hasattr(val, 'item'):  # numpy scalar
        return val.item()
    if isinstance(val, (np.integer, np.floating)):
        return val.item()
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, dict):
        return {sanitize_value(k): sanitize_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return type(val)(sanitize_value(v) for v in val)
    return val


def sanitize_positions(positions):
    """Ensure all position coordinates are native Python floats."""
    return {
        sanitize_value(k): (float(x), float(y)) 
        for k, (x, y) in positions.items()
    }


def main():
    if len(sys.argv) != 3:
        print("Usage: visualize_from_layout.py <layout_file.pkl> <output_file.png>")
        sys.exit(1)

    layout_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    print(f"Loading layout data from {layout_file}...")
    with open(layout_file, 'rb') as f:
        data = pickle.load(f)

    # Sanitize all data to ensure no numpy/pandas types
    positions = sanitize_positions(data['positions'])
    G = data['graph']
    node_interactions = sanitize_value(data['node_interactions'])
    author_handles = data['author_handles']
    show_labels = data['show_labels']
    min_interactions = data['min_interactions']
    min_community_size = data['min_community_size']

    print("Preparing visual attributes...")
    nodes_iter = list(G.nodes())

    # Communities - ensure native Python ints
    communities = [int(sanitize_value(G.nodes[n].get('community', 0))) for n in nodes_iter]

    # Sizes
    sizes_raw = np.array([node_interactions.get(n, 1) for n in nodes_iter])

    if len(sizes_raw) > 0:
        norm = (sizes_raw - sizes_raw.min()) / (sizes_raw.max() - sizes_raw.min() + 1e-9)
        sizes_final = (norm ** 3 * 990 + 10)
    else:
        sizes_final = np.array([10] * len(nodes_iter))

    sizes_list = sizes_final.tolist()

    # Labels
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

    # Draw edges
    nx.draw_networkx_edges(
        G,
        positions,
        alpha=0.1,
        edge_color='gray',
        arrows=False
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        positions,
        node_size=sizes_list,
        node_color=communities,
        cmap=cmap,
        alpha=0.8,
        linewidths=0.5,
        edgecolors='white'
    )

    if show_labels and labels_to_draw:
        nx.draw_networkx_labels(
            G,
            positions,
            labels_to_draw,
            font_size=8,
            font_color='black',
            font_weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7)
        )

    title = f"Community Visualization (cuGraph FA2)\n{G.number_of_nodes()} Nodes"
    if min_interactions:
        title += f" (≥{min_interactions} interactions)"
    if min_community_size:
        title += f"\n(communities with ≥{min_community_size} nodes)"

    plt.title(title, fontsize=16)
    plt.axis('off')

    print(f"Saving to {output_file}...")
    
    # Save as SVG first (Agg/PNG backend has compatibility issues with Pillow)
    svg_path = output_file.with_suffix('.svg')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    print(f"SVG saved to {svg_path}")
    
    # Convert SVG to PNG using cairosvg
    if str(output_file).endswith('.png'):
        try:
            import cairosvg
            cairosvg.svg2png(url=str(svg_path), write_to=str(output_file), scale=4)
            print(f"PNG converted: {output_file}")
            # Keep both files - SVG is useful as vector format
        except ImportError:
            print("Note: cairosvg not installed. Keeping SVG output only.")
            print(f"Install with: pip install cairosvg")
            print(f"Or use the SVG file directly: {svg_path}")
    
    print("Visualization complete!")


if __name__ == "__main__":
    main()
