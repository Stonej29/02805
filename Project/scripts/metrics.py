import networkx as nx


def calculate_edge_betweenness(G, weight='weight'):
    # Convert to undirected if needed
    if G.is_directed():
        G_undir = G.to_undirected()
    else:
        G_undir = G

    edge_betweenness = nx.edge_betweenness_centrality(G_undir, weight=weight)

    return edge_betweenness


def calculate_degree_centrality(G):
    return nx.degree_centrality(G)


def calculate_betweenness_centrality(G, weight='weight'):
    return nx.betweenness_centrality(G, weight=weight)


def calculate_closeness_centrality(G, weight='weight'):
    # For directed graphs, use the weakly connected component
    if G.is_directed():
        if nx.is_weakly_connected(G):
            return nx.closeness_centrality(G, distance=weight)
        else:
            # Calculate for largest weakly connected component
            largest_wcc = max(nx.weakly_connected_components(G), key=len)
            G_wcc = G.subgraph(largest_wcc)
            return nx.closeness_centrality(G_wcc, distance=weight)
    else:
        if nx.is_connected(G):
            return nx.closeness_centrality(G, distance=weight)
        else:
            # Calculate for largest connected component
            largest_cc = max(nx.connected_components(G), key=len)
            G_cc = G.subgraph(largest_cc)
            return nx.closeness_centrality(G_cc, distance=weight)


def calculate_eigenvector_centrality(G, max_iter=1000):
    try:
        return nx.eigenvector_centrality(G, max_iter=max_iter)
    except nx.PowerIterationFailedConvergence:
        return None


def calculate_clustering_coefficient(G):
    if G.is_directed():
        G_undir = G.to_undirected()
        return nx.average_clustering(G_undir)
    else:
        return nx.average_clustering(G)


def calculate_assortativity(G):
    if G.is_directed():
        G_undir = G.to_undirected()
        return nx.degree_assortativity_coefficient(G_undir)
    else:
        return nx.degree_assortativity_coefficient(G)
