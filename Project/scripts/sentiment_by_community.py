import argparse
import json
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import defaultdict
from pathlib import Path
import numpy as np

def load_data(graph_file, posts_file):
    """Load graph and posts"""
    print(f"Loading graph from {graph_file}...")
    G = nx.read_gml(graph_file)
    
    print(f"Loading posts from {posts_file}...")
    posts = []
    with open(posts_file, 'r') as f:
        for line in f:
            try:
                posts.append(json.loads(line))
            except:
                continue
    return G, pd.DataFrame(posts)

def analyze_sentiment(posts_df, G, min_posts=10):
    """Compute sentiment and aggregate by community"""
    print("Computing sentiment...")
    analyzer = SentimentIntensityAnalyzer()
    posts_df['sentiment'] = posts_df['text'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
    
    # Map DIDs to community
    node_community = {}
    comm_nodes = defaultdict(list)
    for node, data in G.nodes(data=True):
        if 'community' in data:
            did = data.get('label', node)
            c = data['community']
            node_community[did] = c
            
            # For leader identification
            degree = G.degree(node)
            handle = data.get('author_handle', 'Unknown')
            comm_nodes[c].append((degree, handle))

    posts_df['community'] = posts_df['author_did'].map(node_community)
    community_posts = posts_df.dropna(subset=['community'])
    community_posts['community'] = community_posts['community'].astype(int)
    
    # Aggregate
    stats = community_posts.groupby('community')['sentiment'].agg(['mean', 'std', 'count'])
    stats = stats[stats['count'] >= min_posts]
    
    # Add metadata (leader, node count, SEM)
    stats['sem'] = stats['std'] / np.sqrt(stats['count'])
    
    leaders = []
    node_counts = []
    for comm_id in stats.index:
        nodes = comm_nodes.get(comm_id, [])
        node_counts.append(len(nodes))
        if nodes:
            _, handle = max(nodes, key=lambda x: x[0])
            leaders.append(handle)
        else:
            leaders.append("Unknown")
            
    stats['leader'] = leaders
    stats['node_count'] = node_counts
    
    return stats

def plot_results(stats, output_dir, top_n=10):
    """Plot mean sentiment with error bars"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort by node count
    top_communities = stats.sort_values('node_count', ascending=False).head(top_n)
    
    labels = [f"C{idx}\n{row['leader'][:15]}..." if len(row['leader']) > 15 else f"C{idx}\n{row['leader']}" 
              for idx, row in top_communities.iterrows()]
    
    plt.figure(figsize=(12, 6))
    # Error bars only (point with error bars)
    plt.errorbar(range(len(top_communities)), top_communities['mean'], yerr=top_communities['sem'], 
                 fmt='o', color='teal', ecolor='black', capsize=5, markersize=8)
            
    plt.xticks(range(len(top_communities)), labels, rotation=45, ha='right')
    plt.title(f'Mean Sentiment by Community (Top {top_n} Largest by Node Count)')
    plt.xlabel('Community')
    plt.ylabel('Mean Sentiment')
    plt.axhline(0, color='black', linestyle='-', linewidth=0.8)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    outfile = output_dir / 'sentiment_mean_by_community.png'
    plt.savefig(outfile)
    plt.close()
    print(f"Plot saved to {outfile}")

def main():
    parser = argparse.ArgumentParser(description='Sentiment Analysis')
    parser.add_argument('--graph', default='../data/bluesky/user_interaction_graph_communities.gml')
    parser.add_argument('--posts', default='../data/bluesky/posts_all.jsonl')
    parser.add_argument('--output', default='../figures/sentiment')
    args = parser.parse_args()
    
    G, posts_df = load_data(args.graph, args.posts)
    stats = analyze_sentiment(posts_df, G)
    
    print("\nTop 10 Communities by Node Size:")
    print(stats.sort_values('node_count', ascending=False).head(10)[['node_count', 'count', 'mean', 'sem', 'leader']])
    
    plot_results(stats, args.output)
    #stats.to_csv(Path(args.output) / 'community_sentiment_stats.csv')

if __name__ == "__main__":
    main()
