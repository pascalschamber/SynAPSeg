# utf-8 encoding
"""
provides several different functions for plotting data

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def determine_feature_uniqueness(X, features, outpath=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ 
    determine auto-correlation of regionprop params - which capture unique information
        plots feature corr matrix dendrogram
    
    Args:
        X: df
        features: list[str]: column names to use as features 
        outpath: str: if provided, save dendrogram figure here
    
    Returns:
        corr: feature autocorrelation matrix
        vif: feature variance_inflation_factor scores 
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from SynAPSeg.utils.utils_plotting import save_fig

    X = X[features].dropna()
    corr = X.corr()

    sns.clustermap(corr, cmap="coolwarm", vmin=-1, vmax=1, figsize=(10,10))
    if outpath: save_fig(outpath)
    plt.show()

    vif = pd.DataFrame()
    vif["feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif.sort_values("VIF"))

    return corr , vif


def plot_ptile_displot(
    pltdf,
    get_ptile_col: str,
    plt_dist_col: str,                  # displot on whatever column you want for x
    group_cols: list = ["treatment"],
    palette=None,
    PTILE = 0.30,                       # default, plots bottom 30, mid 30-70, and top 70 percentiles
    PLOT_MID = True,                    # whether to plot the values between bottom and top PTILES 
    PTILE_FUNCS=None,
    col_wrap = 2,
    **histplot_kwargs
    ) -> pd.DataFrame:
    """
    Filter pltdf by a percentile condition defined in PTILE_FUNCS
        and plot a seaborn displot on the chosen column.
        intended use is to compare side-by-side the bottom and top e.g. 30% of values between treatments

    Args:
        get_ptile_col: col in pltdf used to filter on and create subsets of pltdf to graph in each axis
        plt_dist_col: col in pltdf passed to sns.displot( x=plt_dist_col, ...)
        PTILE: float. 
            Plot bot (x) percentile, mid, and top (1-x) percentile  side-by-side. Only Used if PTILE_FUNCS is None.
        PTILE_FUNCS dict, optional. example: 
            {
                "low_30":  lambda x: x <= x.quantile(0.30),
                "high_30": lambda x: x >= x.quantile(0.70),
            }
    """
    # if ptile_func_key not in PTILE_FUNCS:
    #     raise ValueError(f"Unknown ptile func key: {ptile_func_key}")
    
    if PTILE_FUNCS is None:
        assert isinstance(PTILE, float) 
        if PLOT_MID:
            assert PTILE < 0.50, f"PLOT_MID=True and PTILE={PTILE} not valid. PTILE must be < 0.5 to extract middle dist."

        PTILE_FUNCS = {}
        if PLOT_MID:
            funcStrs = [f"lambda x: x <= x.quantile({PTILE})", f"lambda x: (x > x.quantile({PTILE})) & (x < x.quantile({1-PTILE}))", f"lambda x: x >= x.quantile({1-PTILE})"]
        else:
            funcStrs = [f"lambda x: x <= x.quantile({PTILE})", f"lambda x: x >= x.quantile({1-PTILE})"]

        for get_ptile_funcStr in funcStrs:
            PTILE_FUNCS[get_ptile_funcStr.replace('lambda x: ', f'`{get_ptile_col}`')] = get_ptile_funcStr
    
    # apply funcs to generate subsets of pltdf
    df_percentile = []
    for fname, func in PTILE_FUNCS.items():
        func = eval(func) if isinstance(func, str) else func
        df_percentile.append(
            (
            pltdf
            [pltdf.groupby(group_cols)[get_ptile_col]
            .transform(func)]
            .assign(**{'subset':fname})
            )
        )
    df_percentile = pd.concat(df_percentile, ignore_index=True)

    
    _pltKwargs = dict(
        data=df_percentile,
        x=plt_dist_col,
        kind="hist",
        hue=group_cols[0],
        palette=palette,
        stat="percent",
        element="poly",
        col='subset',
    )
    _pltKwargs.update(**histplot_kwargs)
    
    sns.displot(**_pltKwargs)

    return df_percentile