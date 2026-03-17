"""
Analysis-related Utility functions for dataframes
    for statistics-related analysis see utils_stats
"""

import pandas as pd
import numpy as np

def build_query(**col_values):
    """Build a pandas.DataFrame.query string from keyword pairs.

    Example:
        >>> q = build_query(roi_i=3, region_sides="left")
        >>> df.query(q)

    Args:
        **col_values: Mapping of column names to values used in equality comparisons.

    Returns:
        A string expression suitable for `DataFrame.query`, combining conditions with '&'.
    """
    terms: list[str] = []
    q_val = lambda v: f"'{v}'" if isinstance(v, str) else f"{v}"
    for col, val in col_values.items():
        terms.append(f"({col} == {q_val(val)})")
    return " & ".join(terms)


def query_df(df, **col_values):
    """ helper that wraps df.query(build_query(**col_values))"""
    return df.query(build_query(**col_values))


def ugroups(df, cols) -> pd.DataFrame:
    """ get unique values within columns """
    return df[cols].drop_duplicates()

def filter_present_cols(df, col_list: list[str]):
    """ return list of column in df """
    return [col for col in col_list if col in df.columns]


def get_representative_samples(
    df, 
    value_vars=["count_per_dist", 'roi_intensity_mean_PV'], 
    within_vars=['age'],
    id_vars=["ex_i", "roi_i"], 
    n=None
    ):

    """
    ranks the subjects by thier distance to the group median over all value_vars

    Args:
        df (pd.DataFrame): Input dataframe containing the data.
        value_vars (list|str): The variable to calculate the median for.
        within_vars (list|str): The variable to scale the values within.
        id_vars (list|str): columns which define the `subject`
        n (int): number of representative samples to return
    """
    value_vars = [value_vars] if not isinstance(value_vars, list) else value_vars
    within_vars = [within_vars] if not isinstance(within_vars, list) else within_vars
    id_vars = [id_vars] if not isinstance(id_vars, list) else id_vars
    groupers = id_vars + within_vars

    # melt the value vars to be in long format
    df_long = df.melt(id_vars=groupers, value_vars=value_vars, var_name='value_var', value_name='value').reset_index()

    # scale the values vars so they have equal pull using Z-score normalization (StandardScaler)
    df_long['scaled_value'] = df_long.groupby([*within_vars,'value_var'])['value'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
    )

    # Calculate the per-group median for each value var
    group_medians = df_long.groupby([*within_vars, 'value_var'])['scaled_value'].median().reset_index()
    group_medians.rename(columns={'scaled_value': 'median_val'}, inplace=True)
    merged = pd.merge(df_long, group_medians, on=[*within_vars, 'value_var'])

    # Calculate absolute distance from median for every observation
    merged['dist_to_median'] = (merged['scaled_value'] - merged['median_val']).abs()

    # retain original and scaled values
    representatives = \
        flatten_multindex_df(
        merged.pivot_table(
            columns=['value_var'], 
            index=groupers, 
            values=['value', 'scaled_value'], 
        )
    )

    # Average the distances across all observations (all value_vars and all regions) for each subject
    representative_scores = (
        merged
        .groupby(groupers)['dist_to_median'].mean().reset_index()
        .sort_values('dist_to_median')
        .merge(representatives, on=groupers)
    )
    if n is not None:
        smallest = (
            representative_scores.groupby(within_vars)
            ['dist_to_median'].nsmallest(n)
            .index.get_level_values(-1)
        )
        representative_scores = representative_scores.loc[smallest]
    return representative_scores


def flatten_multindex_df(
    df: pd.DataFrame,
    sep: str = "_",
    drop_empty: bool = True
) -> pd.DataFrame:
    """
    Reset index and flatten MultiIndex columns into single-level string columns.

    Parameters:
        df : pd.DataFrame
            DataFrame with (potentially) MultiIndex index and/or columns
        sep : str, default "_"
            Separator used to join column levels
        drop_empty : bool, default True
            Drop empty / None column levels when flattening

    Returns:
        pd.DataFrame
            DataFrame with reset index and flattened columns
    """
    df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        def flatten(col):
            parts = [
                str(c) for c in col
                if c not in ("", None) or not drop_empty
            ]
            return sep.join(parts)

        df.columns = [flatten(col) for col in df.columns]

    return df
