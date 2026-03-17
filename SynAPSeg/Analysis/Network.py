import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional

# ----------------------------
# Utilities
# ----------------------------
def build_graph_from_corr(
    corr: pd.DataFrame,
    threshold: float = 0.8,
    use_abs: bool = True,
    symmetrize: bool = False,
    zero_diagonal: bool = True,
    drop_nonfinite: bool = True,
) -> nx.Graph:
    """
    Convert a correlation matrix to an undirected weighted graph.
    Edges exist where (abs(corr) >= threshold) if use_abs else (corr >= threshold).

    Args:
        corr: Square DataFrame (index/columns = nodes).
        threshold: Edge threshold.
        use_abs: Use absolute value of correlations for thresholding/weights.
        symmetrize: Force symmetry via (corr + corr.T)/2.
        zero_diagonal: Set diagonal to 0 (no self-loops).
        drop_nonfinite: Replace non-finite with 0 before thresholding.

    Returns:
        NetworkX undirected Graph with 'weight' on edges.
    """
    A = corr.copy()

    if drop_nonfinite:
        A = A.where(np.isfinite(A), 0.0)

    if symmetrize:
        A = (A + A.T) / 2

    if zero_diagonal:
        np.fill_diagonal(A.values, 0.0)

    W = A.abs() if use_abs else A
    A_thr = A.where(W >= threshold, 0.0)

    # networkx treats 0 entries as edges with weight 0 if we pass the full matrix.
    # Safer: build graph only from non-zero edges.
    G = nx.Graph()
    G.add_nodes_from(A_thr.index)

    # Add edges for non-zero weights
    nz = (A_thr != 0)
    rows, cols = np.where(np.triu(nz.values, k=1))
    for i, j in zip(rows, cols):
        w = A_thr.iat[i, j]
        if w != 0:
            G.add_edge(A_thr.index[i], A_thr.columns[j], weight=float(w))

    return G


def _largest_cc(G: nx.Graph) -> nx.Graph:
    """Return the largest connected component subgraph (copy). If G empty, returns empty G."""
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return G.copy()
    if nx.is_connected(G):
        return G.copy()
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


def compute_network_summary(
    G: nx.Graph,
    n_rand: int = 10,
    random_seed: int = 42
) -> Dict[str, float]:
    """
    Compute network-level summary metrics. Uses the largest connected component for path-based stats.

    Returns dict with:
      nodes, edges, density, connected, avg_clustering, avg_path_length,
      small_world_sigma, modularity, assortativity
    """
    summary = dict()
    summary["nodes"] = G.number_of_nodes()
    summary["edges"] = G.number_of_edges()
    summary["density"] = nx.density(G)
    summary["connected"] = float(nx.is_connected(G)) if G.number_of_nodes() > 0 else 0.0

    # Average clustering (weighted if weights exist)
    try:
        summary["avg_clustering"] = nx.average_clustering(G, weight="weight")
    except Exception:
        summary["avg_clustering"] = np.nan

    # Work on GCC for path length / small-worldness
    Gc = _largest_cc(G)
    if Gc.number_of_nodes() >= 2 and Gc.number_of_edges() > 0:
        try:
            L_real = nx.average_shortest_path_length(Gc)  # unweighted geodesic length
        except Exception:
            L_real = np.nan

        try:
            C_real = nx.average_clustering(Gc)
        except Exception:
            C_real = np.nan

        summary["avg_path_length"] = L_real

        # Small-worldness: compare to ER(n, m) graphs
        n, m = Gc.number_of_nodes(), Gc.number_of_edges()
        if n >= 2 and m >= 1 and np.isfinite(L_real) and np.isfinite(C_real):
            rng = np.random.default_rng(random_seed)
            C_rand_list, L_rand_list = [], []
            for _ in range(n_rand):
                Gr = nx.gnm_random_graph(n, m, seed=int(rng.integers(0, 2**31-1)))
                # Ensure Gr is connected for path length comparability; otherwise use its GCC.
                Gr_c = _largest_cc(Gr)
                try:
                    C_rand_list.append(nx.average_clustering(Gr))
                except Exception:
                    C_rand_list.append(np.nan)
                try:
                    L_rand_list.append(nx.average_shortest_path_length(Gr_c))
                except Exception:
                    L_rand_list.append(np.nan)

            C_rand = np.nanmean(C_rand_list) if len(C_rand_list) else np.nan
            L_rand = np.nanmean(L_rand_list) if len(L_rand_list) else np.nan

            if (C_rand and L_rand) and (C_rand > 0) and (L_rand > 0):
                summary["small_world_sigma"] = (C_real / C_rand) / (L_real / L_rand)
            else:
                summary["small_world_sigma"] = np.nan
        else:
            summary["small_world_sigma"] = np.nan
    else:
        summary["avg_path_length"] = np.nan
        summary["small_world_sigma"] = np.nan

    # Modularity via greedy communities (unweighted structure)
    try:
        comms = nx.algorithms.community.greedy_modularity_communities(G)
        summary["modularity"] = nx.algorithms.community.modularity(G, comms)
    except Exception:
        summary["modularity"] = np.nan

    # Assortativity (degree assortativity)
    try:
        summary["assortativity"] = nx.degree_assortativity_coefficient(G)
    except Exception:
        summary["assortativity"] = np.nan

    return summary


def compute_node_metrics(G: nx.Graph) -> pd.DataFrame:
    """
    Compute node-level metrics (all nodes, no truncation):
      degree, strength (weighted degree), degree_centrality,
      eigenvector_centrality, betweenness, k_core, clustering
    """
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return pd.DataFrame(columns=[
            "node", "degree", "strength", "degree_centrality",
            "eigenvector", "betweenness", "k_core", "clustering"
        ])

    # Degree & strength
    degree = dict(G.degree())
    strength = dict(G.degree(weight="weight"))

    # Degree centrality
    deg_c = nx.degree_centrality(G) if len(nodes) > 1 else {n: 0.0 for n in nodes}

    # Eigenvector centrality (weighted); fall back if it fails
    try:
        eig = nx.eigenvector_centrality(G, weight="weight", max_iter=2000)
    except Exception:
        eig = {n: np.nan for n in nodes}

    # Betweenness (remove self-loops just in case; unweighted shortest paths)
    G_no_self = G.copy()
    G_no_self.remove_edges_from(nx.selfloop_edges(G_no_self))
    try:
        btw = nx.betweenness_centrality(G_no_self, normalized=True)
    except Exception:
        btw = {n: np.nan for n in nodes}

    # k-core number
    try:
        core = nx.core_number(G_no_self) if G_no_self.number_of_edges() > 0 else {n: 0 for n in nodes}
    except Exception:
        core = {n: np.nan for n in nodes}

    # Local clustering (weighted if possible)
    try:
        clust = nx.clustering(G, weight="weight")
    except Exception:
        clust = {n: np.nan for n in nodes}

    df = pd.DataFrame({
        "node": nodes,
        "degree": [degree[n] for n in nodes],
        "strength": [strength[n] for n in nodes],
        "degree_centrality": [deg_c[n] for n in nodes],
        "eigenvector": [eig[n] for n in nodes],
        "betweenness": [btw[n] for n in nodes],
        "k_core": [core[n] for n in nodes],
        "clustering": [clust[n] for n in nodes],
    })
    return df


def analyze_groups(
    dfs_byGroupProp,
    prop: str,
    grps: List[str],
    threshold: float = 0.8,
    use_abs: bool = True,
    symmetrize: bool = False,
    zero_diagonal: bool = True,
    drop_nonfinite: bool = True,
    n_rand_smallworld: int = 10,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, nx.Graph]]:
    """
    Run the full pipeline for each group and compile results.

    Args:
        dfs_byGroupProp: dict of dicts holding res.corrs[grp][prop] -> DataFrame (corr matrix).
        prop: which property key to pull (e.g., 'pZif').
        grps: list of group names (keys in res.corrs).
        threshold, use_abs, symmetrize, zero_diagonal, drop_nonfinite: graph construction options.
        n_rand_smallworld: # of ER graphs to average for small-worldness.
        random_seed: RNG seed for reproducibility.

    Returns:
        summary_df: network-level metrics per group (one row/group)
        node_df: node-level metrics with columns [group, node, degree, strength, ...]
        graphs: dict of graphs per group for downstream use
    """
    summaries = []
    node_tables = []
    graphs: Dict[str, nx.Graph] = {}

    for grp in grps:
        corr = dfs_byGroupProp[grp][prop].copy()
        G = build_graph_from_corr(
            corr,
            threshold=threshold,
            use_abs=use_abs,
            symmetrize=symmetrize,
            zero_diagonal=zero_diagonal,
            drop_nonfinite=drop_nonfinite,
        )
        graphs[grp] = G

        # Network-level
        summary = compute_network_summary(G, n_rand=n_rand_smallworld, random_seed=random_seed)
        summary["group"] = grp
        summaries.append(summary)

        # Node-level (all nodes, no top-k)
        node_df = compute_node_metrics(G)
        node_df.insert(0, "group", grp)
        node_tables.append(node_df)

    summary_df = pd.DataFrame(summaries).set_index("group").sort_index()
    node_df = pd.concat(node_tables, ignore_index=True)

    return summary_df, node_df, graphs


# ----------------------------
# Example usage 
# ----------------------------
# prop = 'pZif'
# grps = ["FC", "FC+EXT"]
# threshold = 0.8
# summary_df, node_df, graphs = analyze_groups(
#     res=res,
#     prop=prop,
#     grps=grps,
#     threshold=threshold,
#     use_abs=True,          # typical for correlation networks
#     symmetrize=False,      # set True if your corr isn't perfectly symmetric
#     zero_diagonal=True,    # remove self loops
#     drop_nonfinite=True,   # guard against NaN/inf in corr
#     n_rand_smallworld=20,  # average more ER graphs for stability if you want
#     random_seed=42,
# )
#
# # summary_df: one row per group
# # node_df: all nodes x all node metrics per group
# print(summary_df)
# print(node_df.head())
