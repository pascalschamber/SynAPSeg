# -*- coding: utf-8 -*-
"""
Inter-region property correlation analysis by treatment group.

- Computes per-property region by region Pearson correlations within each treatment.
- Compares treatments with Fisher r-to-z for each region pair.
- BH-FDR correction per property.
- Provides plotting helpers for heatmaps / clustermaps with consistent ordering.

"""
from __future__ import annotations
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # add parent dir to sys.path

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------- Config dataclass -------------------------------

@dataclass
class CorrAnalysisConfig:
    id_col: str = "animal_number"
    group_col: str = "treatment"
    region_col: str = "acronym"
    prop_cols: Optional[Sequence[str]] = None  # default: infer from df
    regions_keep: Optional[Sequence[str]] = None  # subset of regions (e.g., your all_regions)
    min_n_for_test: int = 4  # Fisher z requires n > 3
    corr_method: str = "pearson"  # 'pearson'|'spearman'
    q_method: str = "fdr_bh" # 'fdr_bh' | 'bonferroni'
    # Visualization
    cmap: str = "coolwarm"
    robust: bool = True
    annot: bool = False
    square: bool = True
    figsize: Tuple[int, int] = (7, 6)


# ------------------------------- Core helpers -------------------------------

def _pivot_by_property(
    df: pd.DataFrame,
    cfg: CorrAnalysisConfig,
) -> Dict[str, pd.DataFrame]:
    """
    Returns {property: DataFrame[animals × regions]} for each property.
    Uses pairwise-complete data (pandas.corr default) downstream.
    """
    if cfg.regions_keep is not None:
        df = df[df[cfg.region_col].isin(cfg.regions_keep)].copy()

    # If prop_cols not provided, infer numeric columns excluding IDs
    if cfg.prop_cols is None:
        exclude = {cfg.id_col, cfg.group_col, cfg.region_col}
        num_cols = df.select_dtypes(include=[np.number]).columns
        prop_cols = [c for c in num_cols if c not in exclude]
    else:
        prop_cols = list(cfg.prop_cols)

    # Pivot into MultiIndex columns (prop, region), then split into one matrix per prop
    piv = df.pivot_table(
        index=cfg.id_col,
        columns=cfg.region_col,
        values=prop_cols,
        aggfunc="mean"  # if duplicates per (animal, region), average them
    )
    # `piv` columns are a MultiIndex: (prop, region)
    # Split by first level
    matrices = {}
    for prop in prop_cols:
        if prop in piv.columns.get_level_values(0):
            m = piv[prop]  # animals × regions
            # Drop all-NaN columns (regions that never had values)
            m = m.dropna(axis=1, how="all")
            # Keep only desired regions (again), preserving order if provided
            if cfg.regions_keep is not None:
                keep = [r for r in cfg.regions_keep if r in m.columns]
                m = m.reindex(columns=keep)
            matrices[prop] = m
    return matrices


def _pairwise_counts(mask: pd.DataFrame) -> pd.DataFrame:
    """
    From a boolean mask (animals × regions) of non-null entries,
    compute per-(region_i, region_j) sample counts via mask.T @ mask.
    """
    # Convert to float to use matrix multiply
    A = mask.astype(float)
    counts = A.T @ A  # regions × regions
    counts = counts.astype(int)
    counts.index.name = mask.columns.name
    counts.columns.name = mask.columns.name
    return counts


def _corr_significance(
    X: pd.DataFrame,
    p_method: str = "pearson",
    q_method: str = "fdr_bh",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute pairwise correlation and significance across columns of X.

    Parameters
    ----------
    X : pd.DataFrame
        Rows are samples; columns are brain regions to be correlated.
        Missing values are handled pairwise (listwise for each pair).
    p_method : str, optional
        Correlation method. Currently supports "pearson".
    q_method : str, optional
        Multiple-testing correction method passed to statsmodels.multipletests.
        Common choices: 'fdr_bh' (Benjamini–Hochberg), 'bonferroni', 'holm', etc.

    Returns
    -------
    Dict[str, pd.DataFrame]
        {
            "r":  (n_regions x n_regions) Pearson correlation matrix,
            "p":  (n_regions x n_regions) two-sided p-value matrix,
            "q":  (n_regions x n_regions) multiple-testing-corrected p-values
        }
        All outputs are symmetric DataFrames with index/columns = X.columns.
        Diagonals are r=1, p=0, q=0.
    """
    from scipy.stats import pearsonr
    try:
        from statsmodels.stats.multitest import multipletests
    except Exception as e:
        raise ImportError(
            "statsmodels is required for multiple-comparisons correction. "
            "Install with `pip install statsmodels`."
        ) from e

    if p_method.lower() != "pearson":
        raise ValueError(f"Unsupported p_method='{p_method}'. Only 'pearson' is supported.")

    cols = X.columns.to_list()
    n = len(cols)
    r_mat = np.full((n, n), np.nan, dtype=float)
    p_mat = np.full((n, n), np.nan, dtype=float)

    # Compute upper triangle (excluding diagonal)
    iu, ju = np.triu_indices(n, k=1)
    pvals = np.empty_like(iu, dtype=float)

    for idx, (i, j) in enumerate(zip(iu, ju)):
        xi = X.iloc[:, i]
        xj = X.iloc[:, j]
        # Pairwise dropna for this pair
        valid = xi.notna() & xj.notna()
        if valid.sum() < 3:
            r, p = np.nan, np.nan
        else:
            r, p = pearsonr(xi[valid].values, xj[valid].values)
        r_mat[i, j] = r
        p_mat[i, j] = p
        pvals[idx] = p

    # Mirror to lower triangle; set diagonals
    r_mat[ju, iu] = r_mat[iu, ju]
    p_mat[ju, iu] = p_mat[iu, ju]
    np.fill_diagonal(r_mat, 1.0)
    np.fill_diagonal(p_mat, 0.0)

    # Multiple-comparisons correction on the set of unique tests
    # (upper triangle without diagonal). Keep NaNs as NaN.
    valid_tests = ~np.isnan(pvals)
    qvals = np.full_like(pvals, np.nan, dtype=float)
    if valid_tests.any():
        reject, qvals_valid, _, _ = multipletests(pvals[valid_tests], method=q_method)  # noqa: F841
        qvals[valid_tests] = qvals_valid

    q_mat = np.full((n, n), np.nan, dtype=float)
    q_mat[iu, ju] = qvals
    q_mat[ju, iu] = q_mat[iu, ju]
    np.fill_diagonal(q_mat, 0.0)

    r_df = pd.DataFrame(r_mat, index=cols, columns=cols)
    p_df = pd.DataFrame(p_mat, index=cols, columns=cols)
    q_df = pd.DataFrame(q_mat, index=cols, columns=cols)

    # return {"r": r_df, "p": p_df, "q": q_df}
    return p_df, q_df


def run_correlation_and_stats(
    X: pd.DataFrame, p_method: str = "pearson", q_method: str = 'fdr_bh'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (correlation matrix, pairwise counts matrix, pvals, qvals) for regions in X.
    X is animals × regions.
    """
    corr = X.corr(method=p_method)
    n_ij = _pairwise_counts(X.notna())
    pvals, qvals = _corr_significance(X, p_method=p_method, q_method=q_method)
    return corr, n_ij, pvals, qvals


def _fisher_r_to_z(r: np.ndarray) -> np.ndarray:
    # clip to avoid infinities
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)


def _fisher_compare(r1: np.ndarray, r2: np.ndarray, n1: np.ndarray, n2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fisher r-to-z difference test for two independent correlations.
    Returns (z_diff, p_two_tailed). Only valid where n1>3 and n2>3 and finite r's.
    """
    z1 = _fisher_r_to_z(r1)
    z2 = _fisher_r_to_z(r2)
    se = np.sqrt(1.0 / (np.maximum(n1, 1) - 3) + 1.0 / (np.maximum(n2, 1) - 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (z1 - z2) / se
    p = 2 * (1 - norm.cdf(np.abs(z)))
    # invalid where n too small
    invalid = (n1 < 4) | (n2 < 4) | ~np.isfinite(z) | ~np.isfinite(p)
    z[invalid] = np.nan
    p[invalid] = np.nan
    return z, p


# ------------------------------- Public API -------------------------------

@dataclass
class CorrResults:
    # Per treatment, per property correlation matrices
    corrs: Dict[str, Dict[str, pd.DataFrame]]            # {group: {prop: corr_df}}
    corrs_p: Dict[str, Dict[str, pd.DataFrame]]            # {group: {prop: corr_p_df}} # within groups
    corrs_q: Dict[str, Dict[str, pd.DataFrame]]            # {group: {prop: corr_p_df}}
    counts: Dict[str, Dict[str, pd.DataFrame]]           # {group: {prop: n_ij}}
    # Comparison outputs
    diff_corrs: Dict[str, pd.DataFrame]                  # {prop: (r_group1 - r_group2)} # between groups 
    zscores: Dict[str, pd.DataFrame]                     # {prop: z_diff}
    pvals: Dict[str, pd.DataFrame]                       # {prop: p_two_tailed}
    qvals: Dict[str, pd.DataFrame]                       # {prop: BH-FDR q}
    # Long-format tidy table of significant results
    tidy: pd.DataFrame                                   # columns: [prop, region_i, region_j, r_g1, r_g2, dr, n1, n2, z, p, q]
    group_effects: pd.DataFrame                          # 

def cohens_d(a, b):
    nx, ny = len(a), len(b)
    dof = nx + ny - 2
    return (np.mean(a) - np.mean(b)) / np.sqrt(((nx-1)*np.var(a, ddof=1) + (ny-1)*np.var(b, ddof=1)) / dof)

def analyze_interregion_correlations(
    df: pd.DataFrame,
    group_labels: Tuple[str, str] = ("young", "old"),
    cfg: Optional[CorrAnalysisConfig] = None,
) -> CorrResults:
    """
    Main entry point. Splits df by group, computes per-property region×region correlations,
    compares groups with Fisher r-to-z, and FDR-corrects per property.
    """
    cfg = cfg or CorrAnalysisConfig()

    # df = andf
    # group_labels=("young","old")

    g1, g2 = group_labels
    df1 = df[df[cfg.group_col] == g1].copy()
    df2 = df[df[cfg.group_col] == g2].copy()

    # Build per-property matrices for each group
    mats1 = _pivot_by_property(df1, cfg)
    mats2 = _pivot_by_property(df2, cfg)

    # Align region sets per property (intersection) so matrices are conformable
    props = sorted(set(mats1.keys()).intersection(mats2.keys()))
    corrs = {g1: {}, g2: {}}
    counts = {g1: {}, g2: {}}
    corrs_p = {g1: {}, g2: {}} # pvalues for correlations (within group)
    corrs_q = {g1: {}, g2: {}} # qvalues for correlations (within group)
    diff_corrs, zscores, pvals, qvals = {}, {}, {}, {} # these p and q values are for correlations between groups

    tidy_rows = []
    group_effects = [] # tidy rows for describing significant differences between groups, e.g. not correlations
    for prop in props[:]:
        X1 = mats1[prop]
        X2 = mats2[prop]

        # Intersect regions so we compare same columns
        common_regions = [r for r in X1.columns if r in X2.columns]
        if len(common_regions) < 2:
            # not enough shared regions to compute correlations
            continue

        X1 = X1[common_regions]
        X2 = X2[common_regions]

        # compare group stats on values
        from scipy import stats 
        prop_group_effects = []
        for col in X1.columns:
            _ttest = stats.ttest_ind(X1[col], X2[col], equal_var=False)
            _row = dict(
                prop=prop,
                region=col,
                p = _ttest.pvalue,
                d = cohens_d(X1[col], X2[col]),
                T = _ttest.statistic,
                df = _ttest.df,
                mean_g1 = X1[col].mean(),
                mean_g2 = X2[col].mean(),
                shapiro_g1 = stats.shapiro(X1[col]).pvalue,
                shapiro_g2 = stats.shapiro(X2[col]).pvalue,
                levene = stats.levene(X1[col], X2[col]).pvalue,
            )
            prop_group_effects.append(_row)
        ge = pd.DataFrame(prop_group_effects)
        ge['q'] = multipletests(ge['p'], method=cfg.q_method)[1]
        group_effects.append(ge)


        C1, N1, P1, Q1 = run_correlation_and_stats(X1, p_method=cfg.corr_method, q_method=cfg.q_method)
        C2, N2, P2, Q2 = run_correlation_and_stats(X2, p_method=cfg.corr_method, q_method=cfg.q_method)

        corrs[g1][prop] = C1
        corrs[g2][prop] = C2
        counts[g1][prop] = N1
        counts[g2][prop] = N2
        corrs_p[g1][prop] = P1
        corrs_p[g2][prop] = P2
        corrs_q[g1][prop] = Q1
        corrs_q[g2][prop] = Q2

        Z, P = _fisher_compare(C1.values, C2.values, N1.values, N2.values)
        Z_df = pd.DataFrame(Z, index=common_regions, columns=common_regions)
        P_df = pd.DataFrame(P, index=common_regions, columns=common_regions)

        # Δr = r_g1 - r_g2
        D = C1 - C2
        D = D.reindex(index=common_regions, columns=common_regions)

        # FDR within upper triangle (excluding diagonal) for this property
        iu = np.triu_indices_from(P_df, k=1)
        p_flat = P_df.values[iu]
        valid_mask = np.isfinite(p_flat)
        q_flat = np.full_like(p_flat, np.nan, dtype=float)
        if np.any(valid_mask):
            q_flat[valid_mask] = multipletests(p_flat[valid_mask], method=cfg.q_method)[1]
        Q_df = P_df.copy()
        Q_vals = Q_df.values
        Q_vals[iu] = q_flat
        Q_vals[(iu[1], iu[0])] = q_flat  # mirror to lower triangle
        np.fill_diagonal(Q_vals, np.nan)
        Q_df = pd.DataFrame(Q_vals, index=common_regions, columns=common_regions)

        diff_corrs[prop] = D
        zscores[prop] = Z_df
        pvals[prop] = P_df
        qvals[prop] = Q_df

        # Tidy long table rows for this property (upper triangle only)
        for i, j in zip(*np.triu_indices(len(common_regions), k=1)):
            ri = common_regions[i]
            rj = common_regions[j]
            r1 = C1.iat[i, j]
            r2 = C2.iat[i, j]
            dr = r1 - r2
            n1 = N1.iat[i, j]
            n2 = N2.iat[i, j]
            p1 = P1.iat[i, j]
            p2 = P2.iat[i, j]
            q1 = Q1.iat[i, j]
            q2 = Q2.iat[i, j]
            z = Z_df.iat[i, j]
            p = P_df.iat[i, j]
            q = Q_df.iat[i, j]
            tidy_rows.append(
                {
                    "prop": prop,
                    "region_i": ri,
                    "region_j": rj,
                    f"r_{g1}": r1,
                    f"r_{g2}": r2,
                    "dr": dr,
                    f"n_{g1}": n1,
                    f"n_{g2}": n2,
                    f"p_{g1}": p1,
                    f"p_{g2}": p2,
                    f"q_{g1}": q1,
                    f"q_{g2}": q2,
                    "z": z,
                    "p": p,
                    "q": q,
                }
            )

    tidy = pd.DataFrame(tidy_rows).sort_values(["prop", "q", "p"], na_position="last")

    # group_effects
    tidy_group_effects = pd.concat(group_effects, ignore_index=True).sort_values(["prop", "q", "p"], na_position="last")

    return CorrResults(
        corrs=corrs,
        corrs_p=corrs_p,
        corrs_q=corrs_q,
        counts=counts,
        diff_corrs=diff_corrs,
        zscores=zscores,
        pvals=pvals,
        qvals=qvals,
        tidy=tidy,
        group_effects = tidy_group_effects,
    )


# ------------------------------- Plotting helpers -------------------------------


def plot_group_heatmap(
    corr: pd.DataFrame,
    title: str = "",
    cfg: Optional[CorrAnalysisConfig] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    cfg = cfg or CorrAnalysisConfig()
    if ax is None:
        plt.figure(figsize=cfg.figsize)

    g = sns.heatmap(
        corr, cmap=cfg.cmap, robust=cfg.robust, annot=cfg.annot, square=cfg.square,
        vmin=-1, vmax=1, cbar_kws=dict(label="r"), ax=ax,
    )
    g.set_title(title or "Correlation")
    
    if ax is None:
        plt.tight_layout()
        plt.show()

    return g


def plot_diff_heatmap(
    diff_corr: pd.DataFrame,
    qmask: Optional[pd.DataFrame] = None,
    alpha: float = 0.05,
    title: str = "Δr (group1 - group2)",
    cfg: Optional[CorrAnalysisConfig] = None,
    ax: Optional[plt.Axes] = None,
    **heatmap_kwargs,
):
    """
    Shows Δr heatmap. If qmask is provided (same shape), hatches insignificant cells.
    """
    cfg = cfg or CorrAnalysisConfig()
    if ax is None:
        plt.figure(figsize=cfg.figsize)
    g = sns.heatmap(
        diff_corr,
        cmap=cfg.cmap,
        robust=True,
        annot=cfg.annot,
        square=cfg.square,
        center=0,
        cbar_kws=dict(label="Δr"),
        ax=ax,
        **heatmap_kwargs,
    )
    if qmask is not None:
        # overlay hatch for q>=alpha
        insign = (qmask >= alpha) | ~np.isfinite(qmask)
        for (i, j), val in np.ndenumerate(insign.values):
            if val:
                g.add_patch(
                    plt.Rectangle(
                        (j, i), 1, 1, fill=False, hatch="///", edgecolor="none"
                    )
                )
    g.set_title(title)
    if ax is None:
        plt.tight_layout()
        plt.show()
    return g


def clustermap_with_shared_order(
    corr1: pd.DataFrame, corr2: pd.DataFrame, method: str = "average", metric: str = "euclidean",
    z_score: Optional[int] = None, figsize: Tuple[int, int] = (8, 8), title1: str = "", title2: str = ""
):
    """
    Derive a hierarchical order from corr1 and apply it to corr2 for side-by-side comparison.
    """
    cg1 = sns.clustermap(corr1, method=method, metric=metric, cmap="coolwarm",
                         vmin=-1, vmax=1, figsize=figsize)
    ordered = list(cg1.dendrogram_row.reordered_ind)
    ordered_labels = corr1.index[ordered]
    plt.show()

    corr2o = corr2.reindex(index=ordered_labels, columns=ordered_labels)
    cg2 = sns.clustermap(corr2o, row_cluster=False, col_cluster=False, cmap="coolwarm",
                         vmin=-1, vmax=1, figsize=figsize)
    if title1:
        cg1.ax_heatmap.set_title(title1)
    if title2:
        cg2.ax_heatmap.set_title(title2)
    plt.show()
    return ordered_labels
