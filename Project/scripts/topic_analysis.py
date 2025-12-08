import argparse
import json
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pathlib import Path

def load_graph_and_communities(graph_file):
    """Load graph with community attributes"""
    print(f"Loading graph from {graph_file}...")
    G = nx.read_gml(graph_file)
    
    node_community = {}
    for node, data in G.nodes(data=True):
        if 'community' in data:
            did = data.get('label', node) 
            node_community[did] = data['community']
            
    return node_community

def load_posts(posts_file):
    """Load posts from JSONL"""
    print(f"Loading posts from {posts_file}...")
    posts = []
    with open(posts_file, 'r') as f:
        for line in f:
            try:
                post = json.loads(line)
                posts.append(post)
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(posts)

def analyze_topics(posts_df, node_community, topics=['trump', 'newsom']):
    """Analyze sentiment for topics by community"""
    print("Analyzing topics...")
    analyzer = SentimentIntensityAnalyzer()
    
    # Map posts to communities
    posts_df['community'] = posts_df['author_did'].map(node_community)
    posts_df = posts_df.dropna(subset=['community'])
    posts_df['community'] = posts_df['community'].astype(int)
    
    results = []
    
    for topic in topics:
        print(f"Processing topic: {topic}")
        # Filter posts containing topic
        topic_posts = posts_df[posts_df['text'].str.lower().str.contains(topic, na=False)].copy()
        
        if topic_posts.empty:
            print(f"No posts found for {topic}")
            continue
            
        # Compute sentiment
        topic_posts['sentiment'] = topic_posts['text'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
        
        # Aggregate by community
        stats = topic_posts.groupby('community')['sentiment'].agg(['mean', 'count', 'std'])
        stats['topic'] = topic
        results.append(stats)
        
    if not results:
        return pd.DataFrame()
        
    return pd.concat(results).reset_index()

def plot_polarization(results, output_dir, topics, min_posts=5, community_sizes=None):
    """Plot polarization"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter communities with enough posts
    results_filtered = results[results['count'] >= min_posts]
    
    # Pivot to have topics as columns
    pivot = results_filtered.pivot(index='community', columns='topic', values='mean')
    
    # Check if we have both topics
    if len(topics) == 2 and all(t in pivot.columns for t in topics):
        t1, t2 = topics
        # Keep only communities that have data for BOTH topics
        pivot = pivot.dropna(subset=[t1, t2])
        
        if pivot.empty:
            print("No communities have enough posts for both topics to compare.")
            return

        # Sort
        if community_sizes is not None:
            # Sort by community size (ascending so largest is at top of bar chart)
            pivot['size'] = pivot.index.map(community_sizes)
            pivot = pivot.sort_values('size', ascending=True)
        else:
            # Sort by difference
            pivot['diff'] = pivot[t1] - pivot[t2]
            pivot = pivot.sort_values('diff')
        
        # Plot
        plt.figure(figsize=(12, 8))
        y = np.arange(len(pivot))
        height = 0.35
        
        plt.barh(y - height/2, pivot[t1], height, label=t1.capitalize(), color='red', alpha=0.7)
        plt.barh(y + height/2, pivot[t2], height, label=t2.capitalize(), color='blue', alpha=0.7)
        
        plt.yticks(y, pivot.index)
        plt.xlabel('Mean Sentiment')
        plt.ylabel('Community ID')
        plt.title(f'Sentiment Polarization: {t1.capitalize()} vs {t2.capitalize()} (by Community)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'polarization_{t1}_vs_{t2}.png')
        plt.close()
        
        # Scatter plot
        plt.figure(figsize=(8, 8))
        plt.scatter(pivot[t1], pivot[t2], alpha=0.7, c='purple', edgecolors='w', s=100)
        plt.xlabel(f'{t1.capitalize()} Sentiment')
        plt.ylabel(f'{t2.capitalize()} Sentiment')
        plt.title('Community Sentiment Correlation')
        
        # Fix axes to show full range
        plt.xlim(-1.05, 1.05)
        plt.ylim(-1.05, 1.05)
        plt.axhline(0, color='black', linestyle='-', alpha=0.3)
        plt.axvline(0, color='black', linestyle='-', alpha=0.3)
        
        # Add labels
        for idx, row in pivot.iterrows():
            plt.annotate(str(idx), (row[t1], row[t2]), xytext=(5, 5), textcoords='offset points')
            
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'sentiment_correlation.png')
        plt.close()
    else:
        print(f"Not enough data for both topics ({topics}) to plot comparison.")

def main():
    parser = argparse.ArgumentParser(description='Topic Analysis')
    parser.add_argument('--graph', type=str, default='../data/bluesky/user_interaction_graph_communities.gml')
    parser.add_argument('--posts', type=str, default='../data/bluesky/posts_all.jsonl')
    parser.add_argument('--output', type=str, default='../figures/topics')
    parser.add_argument('--top-n', type=int, default=10, help='Number of top communities to consider')
    parser.add_argument('--topics', nargs='+', default=['trump', 'newsom'], help='Topics to analyze')
    args = parser.parse_args()
    
    # Load data
    node_community = load_graph_and_communities(args.graph)
    
    # Calculate community sizes
    community_counts = pd.Series(list(node_community.values())).value_counts()
    
    # Filter top N communities
    if args.top_n:
        top_communities = community_counts.head(args.top_n).index
        node_community = {n: c for n, c in node_community.items() if c in top_communities}
        print(f"Filtered to top {args.top_n} communities (by node count)")

    posts_df = load_posts(args.posts)
    
    # Analyze
    results = analyze_topics(posts_df, node_community, topics=args.topics)
    
    if results.empty:
        print("No results found.")
        return
        
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot
    plot_polarization(results, output_dir, topics=args.topics, community_sizes=community_counts)
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    main()
