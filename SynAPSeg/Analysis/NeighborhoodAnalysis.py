import ast
from typing import Dict, List, Any, Union
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import seaborn as sns


"""
## **Neighborhood definition**
* Neighborhoods are **local along a dendrite**, so only consider neighbors **on the same `dend_id`**.
* Use **centroid distance** ≤ radius (Euclidean in YX or ZYX) or **K-nearest neighbors** to define neighbors.

**What is measured**
For one or more synapse properties (e.g. area, volume, integrated intensity):
1. For each synapse ( i ):
   * Find all neighbors ( j ) on the **same dendrite**. Exclude self from neighbor list.
2. Compute local neighborhood stats per synapse:
   * `neighbor_mean` = mean(feature of neighbors)
   * `neighbor_std`  = std(feature of neighbors)
   * `n_neighbors`   = number of neighbors
   * etc.


## Usage 
```python
config = {
    "centroid_col": "centroid",
    "dend_id_col": "dend_id",
    "group_col": "age_group", 
    "feature_cols": ["psd95_intensity"],
    "radius_list": [3.0],  # voxel units
    "use_z": True,
    "min_neighbors": 1,
}

data = {"rpdfs": rpdfs}

results = run_synaptic_neighborhood_analysis(data, config)
per_syn = results["per_synapse_stats"]

# Example visualization at radius=3 for psd95_intensity
plot_neighbor_difference_violin(per_syn, feature="psd95_intensity", radius=3.0)
plot_feature_vs_neighbor_mean(per_syn, feature="psd95_intensity", radius=3.0)
```

"""

# -----------------------------
# Low-level helpers
# -----------------------------
def preprocess_NNA_data(
        data: Dict[str, Any],
        config: Dict[str, Any],
):
    """
    Preprocess data['rpdfs'] for synaptic neighborhood analysis
        -drop outliers
        -convert to um coordinate system (from pixels) thru scale features
    """
    ## drop outliers 
    if config['drop_outliers']:
        from SynAPSeg.utils.utils_stats import get_outliers_col

        all_outlier_inds = []
        for c in config['feature_cols']:
            outliers = get_outliers_col(data['rpdfs'], c).index.to_list() 
            all_outlier_inds.extend(outliers)
            print(f"n={len(outliers)} `{c}` outliers")

        all_outlier_inds = list(set(all_outlier_inds))
        print(f"dropping total n={len(all_outlier_inds)} outliers")
        data['rpdfs'] = data['rpdfs'].drop(index=all_outlier_inds)

    ## convert to um coordinate system (from pixels)
    config['scaling'] = config['scaling'] if config['scaling'] else {"Z": 1, "Y": 1, "X": 1}
    scale_arr = np.array([config['scaling'][k] for k in 'ZYX' if k in config['scaling'].keys()])

    scale_coords_fxn = lambda col: col * scale_arr
    data['rpdfs'] [config["centroid_col"]] = (data['rpdfs'] [config["centroid_col"]]
        .apply(_parse_centroid_string)
        .apply(scale_coords_fxn)
    )
    return data

def _parse_centroid_string(s: str) -> np.ndarray:
    """Parse centroid string like '[3.0, 5.1, 10.1]' → np.array."""
    if isinstance(s, (list, tuple, np.ndarray)):
        return np.asarray(s, dtype=float)
    return np.asarray(ast.literal_eval(s), dtype=float)


def ensure_centroid_array_column(
    rpdfs: pd.DataFrame,
    centroid_col: str = "centroid",
    array_col: str = "_centroid_array",
) -> pd.DataFrame:
    """Ensure rpdfs has a column with centroid coordinates as numpy arrays.

    Args:
        rpdfs: Region properties dataframe.
        centroid_col: Name of the string centroid column (e.g. '[z, y, x]').
        array_col: Name of the new column to store np.ndarray centroids.

    Returns:
        rpdfs with an additional column `array_col` containing np.ndarray coords.
    """
    if array_col not in rpdfs.columns:
        rpdfs = rpdfs.copy()
        rpdfs[array_col] = rpdfs[centroid_col].apply(_parse_centroid_string)
    return rpdfs




def compute_neighbor_index_dict(
    rpdfs: pd.DataFrame,
    dend_id_col: str,
    array_col: str = "_centroid_array",
    method: str = "radius",
    radius: float = 2.0,
    k_neighbors: int = 3,
    use_z: bool = True,
) -> Dict[Any, List[Any]]:
    """
    Compute neighbors for each synapse, restricted to each dendrite.
    
    Args:
        rpdfs: Dataframe containing dendrite IDs and centroid arrays.
        dend_id_col: Column name indicating dendrite identity.
        array_col: Column containing np.ndarray centroids.
        method: 'radius' (all within distance) or 'knn' (fixed number of neighbors).
        radius: Neighborhood radius (used if method='radius'), same units as centroid coords.
        k_neighbors: Number of nearest neighbors (used if method='knn').
        use_z: If False, ignore Z dimension (for 2D neighborhood analysis).

    Returns:
        Dictionary mapping central synapse index -> list of neighbor indices.
    """
    neighbor_dict: Dict[Any, List[Any]] = {}
    counter = 0 

    # Group by dendrite to avoid cross-dendrite neighbors
    for dend_id, df_dend in rpdfs.groupby(dend_id_col):

        if len(df_dend) < 2:
            # No neighbors possible on this dendrite
            continue

        # shape (N, Dims)
        coords = np.stack(df_dend[array_col].to_list(), axis=0)

        # Handle dimensionality
        nD = coords.shape[1]
        if nD == 3:  # Stored as [z, y, x]
            use_coords = coords if use_z else coords[:, 1:]
        elif nD == 2:  # Stored as [y, x]
            use_coords = coords
        else:
            raise ValueError(
                f"Unexpected centroid dimensionality: {coords.shape[1]} "
                f"for dend_id={dend_id}"
            )

        # build spaital index
        tree = cKDTree(use_coords)
        global_indices = df_dend.index.to_numpy()

        if method == "radius":
            neighbors_local = tree.query_ball_tree(tree, r=radius)
            # Convert local neighbor indices to global dataframe indices
            for i_local, neigh_local in enumerate(neighbors_local):
                center_idx = global_indices[i_local]
                # Filter out self-match
                neigh_global = [global_indices[j] for j in neigh_local if j != i_local]
                neighbor_dict[center_idx] = neigh_global

        elif method == "knn":
            # We query k+1 because the query point itself will be the closest neighbor (distance 0)
            k_to_query = min(k_neighbors + 1, len(df_dend))
            distances, indices_local = tree.query(use_coords, k=k_to_query)
          

            for i_local, neigh_local in enumerate(indices_local):
                center_idx = global_indices[i_local]
                # tree.query returns a single index if k=1, ensure it's a list
                if k_to_query == 1:
                    neigh_local = [neigh_local]
                
                if counter == 0:
                    print('neigh_local', neigh_local)
                
                # apply max distance thresh (if provided)
                if radius:               
                    neigh_local = np.array(neigh_local)[np.asarray(distances[i_local]) < radius]

                # Filter out self-match and handle potential padding (if k > points)
                neigh_global = [
                    global_indices[j] for j in neigh_local 
                    if j != i_local and j < len(global_indices)
                ]
                neighbor_dict[center_idx] = neigh_global
                counter+=1
        else:
            raise ValueError(f"Unknown method: {method}. Use 'radius' or 'knn'.")
        
        

    return neighbor_dict

def zscore(vals):
    return (vals - vals.mean()) / vals.std()

def compute_feature_neighbor_stats(
    rpdfs: pd.DataFrame,
    feature_col: str,
    neighbor_dict: Dict[int, List[Any]],
    min_neighbors: int = 1,
    dend_id_col = 'dend_id',
    group_col = 'treatment',
    zscore_groupby_cols = ['treatment'],
    n_permutations: int = 100, 
) -> pd.DataFrame:
    """
    Compute neighbor-based statistics for a given feature.
        as built-in tests, also computes neighborhood stats using shuffled values
        - globally shuffled values (randomly shuffled with each group)
        - locally shuffled values (within each dend_id)
        - a Permutation Z-score on neighbors vals relative to randomly sampled 
            vals from the same dend_id
    
    Args:
        rpdfs: Region properties dataframe.
        feature_col: Column name to analyze (e.g. 'syn_intensity').
        neighbor_dict: Mapping from row index → list of neighbor row indices.
        min_neighbors: Minimum neighbors required; others get NaN (or are dropped).
        zscore_groupby_cols: Columns to group by when z-score normalizing the feature. If None, uses the raw feature values.
        n_permutations: Number of permutations to perform for the permutation test.

    Notes:
        For each synapse (row) computes:
            syn_index=idx,
            n_neighbors=n,
            
            # raw values
            feature_value_raw=float(self_val),
            neighbor_mean_raw=float(np.mean(neighbor_vals_raw)),
            neighbor_std_raw=float(np.std(neighbor_vals_raw, ddof=1)) if n > 1 else 0.0,
            neighbor_delta_raw=float(np.mean(neighbor_vals_raw) - self_val),
            
            # Metrics that are now scale-independent, now represent 'Standard Deviations away from the mean'
            feature_value_z=float(self_val_z),
            neighbor_mean_z=float(neighbor_mean_z),
            neighbor_delta_z=float(neighbor_mean_z - self_val_z), 
            local_zscore=z_score,              `neighbor_permutation_zscore` or how different is the neightborhood average from a randomly sampled neighborhood?
            shuffled_neighbor_mean_z,          the mean of the shuffled neighborhood values 
            shuffled_neighbor_delta_z,         the difference between the shuffled neighborhood mean and `self` z scored feature
            
            # single permutation shuffle vals
            shuffled_mean_global=float(np.mean(shuffled_global_vals)),
            shuffled_std_global=float(np.std(shuffled_global_vals, ddof=1)) if n > 1 else 0.0,
            shuffled_mean_local=float(np.mean(shuffled_local_vals)),
            shuffled_std_local=float(np.std(shuffled_local_vals, ddof=1)) if n > 1 else 0.0,

    Returns:
        DataFrame with one row per synapse (subset of rpdfs.index)
            and columns: ['syn_index', 'feature_value', 'neighbor_mean',
                        'neighbor_std', 'n_neighbors', 'neighbor_shuffled_mean',
                        'neighbor_shuffled_std', 'neighbor_permutation_zscore']
            note: row indicies of the input dataframe (rpdfs) are preserved in the `syn_index` column
    """
    if zscore_groupby_cols is not None: # uses the zscored feature values for permutation calculations
        z_feature_col = f"z_{feature_col}"
        rpdfs[z_feature_col] = rpdfs.groupby(zscore_groupby_cols)[feature_col].transform(
            lambda x: zscore(x)
        )
    else:
        z_feature_col = feature_col
    
    # Pre-calculate single shuffles
    global_shuffled_vals = (
        rpdfs.groupby(group_col)[z_feature_col]
        .transform(lambda x: x.sample(frac=1).values)
    )
    local_shuffled_vals = (
        rpdfs.groupby(dend_id_col)[z_feature_col]
        .transform(lambda x: x.sample(frac=1).values)
    )
    
    # Use the standardized values for the neighborhood logic
    vals = rpdfs[z_feature_col] 
    raw_vals = rpdfs[feature_col] # Keep raw for the final record
    
    # Pre-group values by dendrite to speed up the permutation loop
    dend_groups = rpdfs.groupby(dend_id_col)[z_feature_col].apply(list).to_dict()
    synapse_to_dendrite = rpdfs[dend_id_col].to_dict()

    records = []
    for idx, neighbors in neighbor_dict.items():

        # get neighborhood values, skip if too few
        ###################################################################################################
        neighbor_vals = vals.loc[neighbors].to_numpy(dtype=float) # z-scored neighborhood values
        n = neighbor_vals.size
        
        if n < min_neighbors:
            continue

        # get raw values for final record
        ###################################################################################################
        self_val            = raw_vals.loc[idx] # raw `self` feature value
        neighbor_vals_raw   = raw_vals.loc[neighbors].to_numpy(dtype=float) # raw mean of neighbors
            
        # single permutation shuffles
        shuffled_global_vals = global_shuffled_vals.loc[neighbors].to_numpy(dtype=float)
        shuffled_local_vals = local_shuffled_vals.loc[neighbors].to_numpy(dtype=float)
        
        # --- MULTI-ITERATION PERMUTATION TEST (Z-SCORE) ---
        ###################################################################################################
        # Get all possible feature values on this specific dendrite
        pool = dend_groups[synapse_to_dendrite[idx]]
        self_val_z      = vals.loc[idx] # z-scored `self` feature value
        neighbor_mean_z = np.mean(neighbor_vals) # z-scored mean of neighbors
        neighbor_std_z: float = np.std(neighbor_vals, ddof=1) if n > 1 else 0.0  
        
        if (n_permutations > 0) and (len(pool) > n):
            
            # Randomly pick 'n' values from the dendrite pool n_permutations times
            null_means = []
            for _ in range(n_permutations):
                resampled = np.random.choice(pool, size=n, replace=False)
                null_means.append(np.mean(resampled))
            
            shuffled_neighbor_mean_z = np.mean(null_means)
            
            # Calculate the z-score of the neighborhood mean vs random neighborhood means
            null_sd = np.std(null_means)
            local_zscore = (neighbor_mean_z - shuffled_neighbor_mean_z) / null_sd if null_sd > 0 else 0.0

        else: # Not enough synapses on dendrite to shuffle
            local_zscore = np.nan 
            shuffled_neighbor_mean_z = np.nan

        # agg results for this synapse and neighborhood
        ###################################################################################################
        records.append(
            dict(                           # type: ignore[bad-argument-type]
                syn_index=idx,
                n_neighbors=n,
                # raw values
                feature_value_raw=float(self_val),
                neighbor_mean_raw=float(np.mean(neighbor_vals_raw)),
                neighbor_std_raw=float(np.std(neighbor_vals_raw, ddof=1)) if n > 1 else 0.0,
                neighbor_delta_raw=np.nan,
                # Metrics that are now scale-independent
                feature_value_z=float(self_val_z),
                neighbor_mean_z=float(neighbor_mean_z),
                neighbor_std_z = float(neighbor_std_z),
                neighbor_delta_z=np.nan,
                shuffled_neighbor_mean_z = float(shuffled_neighbor_mean_z),
                local_zscore=local_zscore,
                shuffled_neighbor_delta_z=np.nan,
                # single permutation shuffle vals
                shuffled_mean_global=float(np.mean(shuffled_global_vals)),
                shuffled_std_global=float(np.std(shuffled_global_vals, ddof=1)) if n > 1 else 0.0,
                shuffled_delta_global=np.nan,
                shuffled_mean_local=float(np.mean(shuffled_local_vals)),
                shuffled_std_local=float(np.std(shuffled_local_vals, ddof=1)) if n > 1 else 0.0,
                shuffled_delta_local=np.nan,
            )
        )

    df = pd.DataFrame.from_records(records)
    
    # efficent calculation of neighbor differences
    df['neighbor_delta_raw'] = df['neighbor_mean_raw'] - df['feature_value_raw']
    df['neighbor_delta_z'] = df['neighbor_mean_z'] - df['feature_value_z']
    df['shuffled_neighbor_delta_z'] = df['shuffled_neighbor_mean_z'] - df['feature_value_z']
    df['shuffled_delta_global'] = df['shuffled_mean_global'] - df['feature_value_raw']
    df['shuffled_delta_local'] = df['shuffled_mean_local'] - df['feature_value_raw']

    return df

# -----------------------------
# API layer / orchestration
# -----------------------------
def run_synaptic_neighborhood_analysis(
    data: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """High-level API to run synaptic neighborhood analysis.

    Args:
        data: Dictionary containing 'rpdfs' DataFrame.
        config: Dictionary with analysis parameters:
            - 'scaling' (Dict[str, float]): scaling factors for each dimension (during preprocessing), 
                defaults to {"Z": 1, "Y": 1, "X": 1}.
            - 'method' (str): "radius" or "knn".
            - 'radius_list' (List[float]): Distances to test (if method="radius").
            - 'k_list' (List[int]): N-neighbors to test (if method="knn").
            - 'centroid_col' (str): Centroid column name.
            - 'dend_id_col' (str): Dendrite ID column.
            - 'feature_cols' (List[str]): Features to analyze.
            - 'use_z' (bool): Include Z dimension.
            - 'min_neighbors' (int): Minimum neighbors required to keep a row. The value passed here is ignored if using knn method
            - 'group_col' (str or None): group column (e.g. 'age_group').
            - 'n_permutations' (int): Number of permutations for z-score neighborhood featurepermutation test.

    Returns:
        results dict with:
            - 'per_synapse_stats': DataFrame with per-synapse neighborhood metrics.
            - 'config_used': copy of the config.
            - 'nNeighbors_distribution': DataFrame summarizing the distribution of neighbors at each method threshold.
    """

    data = preprocess_NNA_data(data, config)    
    rpdfs: pd.DataFrame = data["rpdfs"]

    # Extract config parameters with defaults
    method = config.get("method", "radius")
    centroid_col = config.get("centroid_col", "centroid")
    dend_id_col = config.get("dend_id_col", "dend_id")
    group_col = config.get("group_col", None)

    feature_cols = config.get("feature_cols", [])
    use_z = config.get("use_z", True)
    min_neighbors = config.get("min_neighbors", 1)
    n_permutations = config.get("n_permutations", 100)
    zscore_groupby_cols = config.get("zscore_groupby_cols", group_col)

    # Determine which parameter set to iterate over
    if method == "knn":
        param_list = config.get("k_list", [3])
        param_name = "k_neighbors"
        max_distance = config.get("k_max_dist") or None
        

    else:
        param_list = config.get("radius_list", [2.0])
        param_name = "radius"
        

    if not feature_cols:
        raise ValueError("config['feature_cols'] must contain at least one feature name.")

    # Ensure we have array centroids
    rpdfs = ensure_centroid_array_column(rpdfs, centroid_col=centroid_col)
    all_stats = []

    # Iterate through different neighborhood sizes (either radius or K)
    for val in param_list:
        # Define arguments for neighbor dict computation
        neighbor_args = {
            "rpdfs": rpdfs,
            "dend_id_col": dend_id_col,
            "array_col": "_centroid_array",
            "method": method,
            "use_z": use_z,
            param_name: val,                # Dynamically assign the method parameter
        }
        
        # pass radius to use for maximum distance
        if method == "knn":
            neighbor_args['radius'] = max_distance 
            
        neighbor_dict = compute_neighbor_index_dict(**neighbor_args)
        if not neighbor_dict:
            continue

        for feature in feature_cols:
            feature_stats = compute_feature_neighbor_stats(
                rpdfs=rpdfs,
                feature_col=feature,
                neighbor_dict=neighbor_dict,
                min_neighbors=val if method=='knn' else min_neighbors,
                dend_id_col = dend_id_col,
                group_col = group_col,
                n_permutations = n_permutations,
                zscore_groupby_cols = zscore_groupby_cols,
            )

            if feature_stats.empty:
                continue

            # Attach metadata: dend_id, group label, radius, feature

            feature_stats["dend_id"] = rpdfs.loc[
                feature_stats["syn_index"], dend_id_col
            ].values

            if group_col is not None and group_col in rpdfs.columns:

                feature_stats[group_col] = rpdfs.loc[
                    feature_stats["syn_index"], group_col
                ].values

            else:
                feature_stats[group_col] = "all"

            # Log the specific neighborhood parameter used
            feature_stats["neighborhood_method"] = method
            feature_stats[param_name] = val
            feature_stats["feature"] = feature

            all_stats.append(feature_stats)
    
    if all_stats:
        per_synapse_stats = pd.concat(all_stats, ignore_index=True)

        grpers = ([] if not group_col else [group_col]) + [param_name] 
        
        # validate/review number of neighbors at each method (e.g. radius or k_neighbors) threshold
        percentiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        nNeighbors_distribution = (
            per_synapse_stats
            .groupby(grpers, as_index=False)
            .agg({'n_neighbors': [lambda x, q=q: x.quantile(q) for q in percentiles]}))
        mapping = {f'<lambda_{i}>': f'p{p}' for i, p in enumerate(percentiles)}
        nNeighbors_distribution = nNeighbors_distribution.rename(columns=mapping, level=1)

        # calculate the proportion of clusters defined by N neighbors
        nNeighbors_counts = (
            per_synapse_stats
            .groupby(grpers)
            ['n_neighbors']
            .apply(lambda s: s.value_counts(normalize=True)) 
            .unstack(fill_value=0) # Pivot unique neighbor counts into columns
            .reset_index()
            .sort_values(grpers[::-1])
        )
                
    else:
        per_synapse_stats = pd.DataFrame()
        nNeighbors_distribution = pd.DataFrame()
        nNeighbors_counts = pd.DataFrame()


    return {
        "per_synapse_stats": per_synapse_stats,
        "config_used": dict(config),
        "nNeighbors_distribution": nNeighbors_distribution,
        "nNeighbors_counts": nNeighbors_counts,
    }


# -----------------------------
# Visualization helpers
# -----------------------------

def plot_neighbor_difference_violin(
    per_synapse_stats: pd.DataFrame,
    feature: str,
    y_col: str,                     # e.g. 'neighbor_delta_z', 'neighbor_delta'
    neighbor_method='radius', 
    neighborhood_threshold=2,
    group_col: str = "age_group",
    figsize=(6, 4),
):
    """Plot distribution of (neighbor_mean - feature_value) across groups.

    Args:
        per_synapse_stats: Output from run_synaptic_neighborhood_analysis()['per_synapse_stats'].
        feature: Feature name to visualize (must match 'feature' column).
        radius: Neighborhood radius to subset.
        group_col: Group column in per_synapse_stats (e.g. 'age_group').
        figsize: Figure size.
    """
    df = per_synapse_stats
    mask = (df["feature"] == feature) & (df[neighbor_method] == neighborhood_threshold)
    df_sub = df.loc[mask].copy()

    if df_sub.empty:
        print("No data to plot for this feature and radius.")
        return

    plt.figure(figsize=figsize)
    sns.violinplot(
        data=df_sub,
        x=group_col,
        y=y_col,
        inner="box",
        cut=0,
    )
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(f"Neighbor Δ ({feature}) at {neighbor_method}={neighborhood_threshold}")
    plt.ylabel("neighbor_mean - feature_value")
    plt.xlabel(group_col)
    plt.tight_layout()
    plt.show()


def plot_feature_vs_neighbor_mean(
    per_synapse_stats: pd.DataFrame,
    feature: str,
    neighbor_method='radius', 
    neighborhood_threshold=2,
    group_col: str = "age_group",
    figsize=(5, 5),
    add_unity_line: bool = True,
):
    """Scatter/regression plot of feature_value vs neighbor_mean by group.

    Args:
        per_synapse_stats: Output from run_synaptic_neighborhood_analysis()['per_synapse_stats'].
        feature: Feature name to visualize.
        radius: Neighborhood radius to subset.
        group_col: Group column (e.g. 'age_group').
        figsize: Figure size.
        add_unity_line: If True, draw y = x line.
    """
    df = per_synapse_stats
    mask = (df["feature"] == feature) & (df[neighbor_method] == neighborhood_threshold)
    df_sub = df.loc[mask].copy()

    if df_sub.empty:
        print("No data to plot for this feature and radius.")
        return

    plt.figure(figsize=figsize)
    sns.scatterplot(
        data=df_sub,
        x="feature_value",
        y="neighbor_mean",
        hue=group_col,
        alpha=0.5,
    )
    if add_unity_line:
        lims = [
            min(df_sub["feature_value"].min(), df_sub["neighbor_mean"].min()),
            max(df_sub["feature_value"].max(), df_sub["neighbor_mean"].max()),
        ]
        plt.plot(lims, lims, linestyle="--", linewidth=1)
        plt.xlim(lims)
        plt.ylim(lims)

    plt.title(f"{feature}: self vs neighbor mean ({neighbor_method}={neighborhood_threshold})")
    plt.xlabel("Synapse feature value")
    plt.ylabel("Mean of neighbor feature values")
    plt.tight_layout()
    plt.show()


def plot_clustering_zscores(stats_df, treatment_col='treatment'):
    """
    Plots the distribution of local Z-scores by treatment group.
    
    Args:
        stats_df: The output from compute_feature_neighbor_stats 
                  (must be merged with treatment metadata).
    """
    plt.figure(figsize=(8, 6))
    
    # 1. Create the Violin Plot
    sns.violinplot(
        data=stats_df, 
        x=treatment_col, 
        y='local_zscore', 
        inner='quartile', 
        palette='muted',
        alpha=0.7
    )
    
    # 2. Add a horizontal line at Z=0 (the null hypothesis)
    plt.axhline(0, color='black', linestyle='--', linewidth=1, label='Random Distribution')
    
    # 3. Add significance thresholds (Z > 1.96 is p < 0.05)
    plt.axhline(1.96, color='red', linestyle=':', alpha=0.5, label='Significance (p < 0.05)')
    
    # Clean up labels
    plt.title('Synaptic Clustering Rigor: Local Z-Scores by Treatment', fontsize=14)
    plt.ylabel('Neighbor Intensity Z-Score (vs. Local Null)', fontsize=12)
    plt.xlabel('Treatment Group', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1))
    
    sns.despine() # Clean academic look
    plt.tight_layout()
    
