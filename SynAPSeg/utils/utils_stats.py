import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import pandas as pd
from tabulate import tabulate
from enum import Enum
import numpy as np


def batch_indSamples_test(df, value_cols:list[str], groupby_cols:list[str], treatment_col='treatment',  group1=None, group2=None):
    """ 
    unifying interface for batch running t-tests and mwu tests to compare two groups 

    Args:
        df (pd.DataFrame): 
        value_cols (list[str]): columns to run tests on
        groupby_cols (list[str]): columns to group by for each test
        treatment_col (str, optional): groups to compare.
        group1 (str, optional): Defaults to None.
        group2 (str, optional): Defaults to None.
    
    """
    from . import utils_general as ug
    
    def _run_stats(_df, value_cols:list[str], treatment_col='treatment',  group1=None, group2=None):
        
        results = []
        for fc in value_cols:
            res = mwu_run(_df, fc, grp1=group1, grp2=group2, treatment_col=treatment_col)
            comp_res = {'variable':fc, 'mwu_statistic':res.statistic, 'mwu_pvalue':res.pvalue}
            res_ttest = check_ttest_assumptions_and_run(_df, fc, treatment_col=treatment_col, group1=group1, group2=group2, plots=False)
            flat_res = ug.flatten_dict(res_ttest)
            comp_res['t-test_p-value'] = flat_res.pop('t-test_p-value')
            comp_res['t-test_pass'] = flat_res.pop('t-test_pass')
            comp_res.update(flat_res)

            # add group labels and Ns
            g1_n = len(_df.loc[_df[treatment_col] == group1, fc].dropna())
            g2_n = len(_df.loc[_df[treatment_col] == group2, fc].dropna())
            comp_res.update({
                'group1':group1, 'N1':g1_n, 'group2':group2, 'N2':g2_n
            })
            results.append(comp_res)

        return pd.DataFrame(results)

    group1, group2 = interpret_groups_twoSided(df, group1, group2, treatment_col=treatment_col)
    
    if not groupby_cols:
        return _run_stats(df, value_cols, treatment_col, group1, group2) 

    all_res = df.groupby(groupby_cols).apply(
        lambda group: _run_stats(group, value_cols, treatment_col, group1, group2)
    ).reset_index()
    
    return all_res


class Significance(Enum):
    P0001 = ("****", 0.0001) # order is important
    P001  = ("***",  0.001)
    P01   = ("**",   0.01)
    P05   = ("*",    0.05)
    NS    = ("",     float("inf")) 

    @property
    def stars(self):
        return self.value[0]

    @property
    def threshold(self):
        return self.value[1]

    @classmethod
    def from_pvalue(cls, p):
        # iterate in the order defined above
        for level in cls:
            if p < level.threshold:
                return level
        # fallback, though we should always hit NS
        return cls.NS

    @classmethod
    def _missing_(cls, p):
        """ 
        makes class callable with non-explicit values, returning cls.threshold < pval
        
        Examples:
            Significance(0.000001) --> <Significance.P0001: ('****', 0.0001)>
            Significance(0.000001).stars --> `****`
        
        """
        for level in cls:
            if p < level.threshold:
                return level
        return cls.NS

def sig_rep(pval):
    return Significance.from_pvalue(pval).stars

def fmt_pval(p):
    if p > 0.05: return round(p,2)
    else: return f"{p:.2e}"





def summarize_batch_stats(df, tablefmt="fancy_grid"):
    # TODO multiple comparisons correction
    def rnd(pval):
        return round(pval, 4)

    rows = []
    for ri, row in df.iterrows():
        ttestpassed = "passed" if row["t-test_pass"] else ""
        ttp = row["t-test_p-value"]
        mwup = row["mwu_pvalue"]
        anysig = sig_rep(min((ttp if row["t-test_pass"] else 1.0, mwup)))

        rows.append([
            f"{row['variable']} {anysig}",
            ttestpassed,
            rnd(ttp),
            sig_rep(ttp),
            rnd(mwup),
            sig_rep(mwup),
        ])

    headers = [
        "variable",
        "t-test",
        "t-test p",
        "",
        "MWU p",
        "",
    ]

    print(tabulate(rows, headers=headers, tablefmt=tablefmt))

def interpret_groups_twoSided(df, grp1=None, grp2=None, treatment_col='treatment'):
    """ helper function to automatically determine group values in treatment column if only 2 unique vals """
    if grp1 is None or grp2 is None:
        grps = sorted(df[treatment_col].unique())
        if len(grps) != 2:
            raise ValueError((
                "if treatment groups not provided, must be exactly 2 unique values in the treatment column\n",
                f"\tfound n={len(grps)} in sorted(df[{treatment_col}].unique()): {grps}"
            ))
        grp1, grp2 = grps
    
    # if both not none, just returns them
    return grp1, grp2

def mwu_run(df, feature, grp1=None, grp2=None, treatment_col='treatment'):
    """"""
    grp1, grp2 = interpret_groups_twoSided(df, grp1=grp1, grp2=grp2, treatment_col=treatment_col)

    g1 = df.loc[df[treatment_col] == grp1, feature]
    g2 = df.loc[df[treatment_col] == grp2, feature]

    return scipy.stats.mannwhitneyu(g1, g2, alternative="two-sided")

def batch_mwu_run(df, cols, grp1=None, grp2=None, treatment_col='treatment'):
    mwu_results = []
    for fc in cols:
        res = mwu_run(df, fc, grp1=grp1, grp2=grp2, treatment_col=treatment_col)
        mwu_results.append({'variable':fc, 'mwu_statistic':res.statistic, 'mwu_pvalue':res.pvalue})
    return pd.DataFrame(mwu_results)

def merge_mwu_with_ttest_run(ttestDf, mwu_results):
    return ttestDf.merge(mwu_results, how='outer', on='variable')

def batch_ttest_run(df, y_var, iter_col, treatment_col='treatment',  group1=None, group2=None):
    """ runs check_ttest_assumptions_and_run over unique values in iter_col """
    from . import utils_general as ug
    uitervals = df[iter_col].unique()

    group1, group2 = interpret_groups_twoSided(df, group1, group2, treatment_col=treatment_col)

    all_res = {v:{} for v in uitervals}
    for uval in uitervals:
        _df = get_ars(df, None, **{iter_col:uval})
        if len(_df)==0:
            continue
        try:
            res = check_ttest_assumptions_and_run(_df, y_var, treatment_col=treatment_col, group1=group1, group2=group2, plots=False)
        except Exception as e:
            print(e)
            continue
        flat_res = ug.flatten_dict(res)
        all_res[uval] = flat_res
    return pd.DataFrame(data=[all_res[k] for k in all_res.keys()], index=list(all_res.keys())).reset_index().rename(columns={'index':iter_col}).assign(**{'var':y_var})

def get_ars(df, col=None, **filters):
    """
    Filter a DataFrame by arbitrary column-value pairs and optionally return a specific column.

    Args:
        df (pd.DataFrame): The input DataFrame to filter.
        col (str, optional): Name of the column to return. If None, the filtered DataFrame is returned.
        **filters: Arbitrary keyword arguments specifying column-value pairs to filter on. 
            Values can be of any type supported by `pandas.DataFrame.query`.

    Returns:
        pd.DataFrame or pd.Series: 
            - If `col` is None, returns the filtered DataFrame.
            - If `col` is specified, returns the corresponding column (as a Series) from the filtered DataFrame.

    Example:
        get_ars(df, age='P60', reg_id=123)
        get_ars(df, col='nuclei_count', group='control', brain_region='CA1')
    """
    if filters:
        query_str = " & ".join(
            [f"{k} == '{v}'" if isinstance(v, str) else f"{k} == {v}" for k, v in filters.items()]
        )
        res = df.query(query_str)
    else:
        res = df

    return res if col is None else res[col]


def check_ttest_assumptions_and_run(df, y_var, treatment_col='treatment', group1='KO', group2='control', plots=True):
    """
    This function checks the assumptions of a t-test and performs the t-test
    if all assumptions are met.

    Parameters:
    - df: DataFrame containing the data
    - y_var: The dependent variable to test
    - treatment_col: The column name representing the treatment groups (default 'treatment')
    - group1: The first group to compare (default 'KO')
    - group2: The second group to compare (default 'control')
    - plots: if True shows Q-Q and histogram

    Returns:
    - A dictionary with the results of the tests and whether assumptions are met.
    """
    
    # Define the two groups based on treatment column
    group1_data = df[df[treatment_col] == group1][y_var]
    group2_data = df[df[treatment_col] == group2][y_var]

    results = {}

    # Step 1: Test for normality using the Shapiro-Wilk test
    normality_group2 = scipy.stats.shapiro(group2_data)
    normality_group1 = scipy.stats.shapiro(group1_data)

    results['shapiro_group1'] = {'W': normality_group1[0], 'p-value': normality_group1[1]}
    results['shapiro_group2'] = {'W': normality_group2[0], 'p-value': normality_group2[1]}

    # print(f"Shapiro-Wilk Test for {group1} group: W={normality_group1[0]}, p-value={normality_group1[1]}")
    # print(f"Shapiro-Wilk Test for {group2} group: W={normality_group2[0]}, p-value={normality_group2[1]}")

    # Step 2: Test for homogeneity of variances using Levene's test
    levene_test = scipy.stats.levene(group1_data, group2_data)
    results['levene'] = {'W': levene_test[0], 'p-value': levene_test[1]}

    # print(f"Levene's Test for equal variances: W={levene_test[0]}, p-value={levene_test[1]}")

    if plots:
        # Step 3: Visualize the data with Q-Q plots for normality
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        scipy.stats.probplot(group1_data, dist="norm", plot=plt)
        plt.title(f'Q-Q Plot for {group1} group')

        plt.subplot(1, 2, 2)
        scipy.stats.probplot(group2_data, dist="norm", plot=plt)
        plt.title(f'Q-Q Plot for {group2} group')

        plt.show()

        # Step 4: Visualize the distributions using histograms
        plt.figure(figsize=(10, 5))
        sns.histplot(group1_data, kde=True, label=group1, color='blue', stat="density")
        sns.histplot(group2_data, kde=True, label=group2, color='red', stat="density")
        plt.title(f'Histogram of {group1} and {group2} Groups')
        plt.legend()
        plt.show()

    # Step 5: If assumptions are met, perform the t-test
    if normality_group1[1] > 0.05 and normality_group2[1] > 0.05 and levene_test[1] > 0.05:
        t, p = scipy.stats.ttest_ind(group1_data, group2_data)
        results['t-test_pass'] = True
        results['t-test'] = {'t': t, 'p-value': p}
        # print(f"T-test: t={t}, p-value={p}")
    else:
        # print("T-test assumptions not met.")
        t, p = scipy.stats.ttest_ind(group1_data, group2_data)
        # print(f"T-test: t={t}, p-value={p}")
        results['t-test_pass'] = False
        results['t-test'] = {'t': t, 'p-value': p}


    return results



def get_ttest_stats_summary(ttest_stats):
    res = 'pass' if ttest_stats['t-test_pass'] else 'fail'
    return f"ttest ({res}): t={ttest_stats['t-test']['t']:.2f}, p={fmt_pval(ttest_stats['t-test']['p-value'])}"

def compare_groups(df, value_col, group_col, test='anova'):
    """
    Compare groups using ANOVA or Kruskal-Wallis H-test.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    value_col (str): Column name for the values.
    group_col (str): Column name for the group labels.
    test (str): The type of test to perform, 'anova' for ANOVA and 'kruskal' for Kruskal-Wallis.
    
    Returns:
    float, float: Test statistic and p-value.
    """
    
    # Get unique groups
    groups = df[group_col].unique()
    
    # Split the data into groups
    group_data = [df[df[group_col] == group][value_col] for group in groups]
    
    if test == 'anova':
        # Check normality assumption using Shapiro-Wilk test for each group
        normality = all(scipy.stats.shapiro(group)[1] > 0.05 for group in group_data)
        
        # Check homogeneity of variances using Levene's test
        homogeneity = scipy.stats.levene(*group_data)[1] > 0.05
        
        if normality and homogeneity:
            # Perform ANOVA
            f_stat, p_val = scipy.stats.f_oneway(*group_data)
            test_name = "ANOVA"
        else:
            raise ValueError("Data does not meet the assumptions for ANOVA. Consider using Kruskal-Wallis test.")
    
    elif test == 'kruskal':
        # Perform Kruskal-Wallis H-test
        f_stat, p_val = scipy.stats.kruskal(*group_data)
        test_name = "Kruskal-Wallis H-test"
    else:
        raise ValueError("Invalid test type specified. Use 'anova' or 'kruskal'.")
    
    return test_name, f_stat, p_val


def get_corr_stats(pltdf, xcol, ycol, groupers):
    
    stats_result = []
    for dfn, adf in pltdf.groupby(groupers):
        x,y = adf[xcol].values, adf[ycol].values
        _stats = dict(
            pear_corr = scipy.stats.pearsonr(x,y),
            spear_corr = scipy.stats.spearmanr(x,y)
        )
        _stats_result = {
                **dict(zip(groupers, dfn)), **{'N': len(x)},
            }
        for sname, s in _stats.items():
            _stats_result[f"{sname}_stat"] = s.statistic
            _stats_result[f"{sname}_p"] = s.pvalue
        stats_result.append(_stats_result)
    corr_stats = pd.DataFrame(stats_result)

    return corr_stats, get_corr_stats_summary(corr_stats, groupers)






def compare_correlations_stats(pltdf:pd.DataFrame, corr_cols:list[str], grouppers:list[str]):
    """
    compute correlations for each group in groupers over specified columns and run a fisher r-to-z test to compare

    returns:
        corr_stats (pd.DataFrame): correlation stats for each group
        sts_summary (str): summary of correlation stats
    """
    from scipy.stats import pearsonr
    from scipy.stats import spearmanr

    assert len(corr_cols) == 2
    assert len(grouppers) > 0
    
    stats_result = []
    for dfn, adf in pltdf.groupby(grouppers):
        x,y = [adf[cc].values for cc in corr_cols]
        _stats = dict(
            pear_corr = pearsonr(x,y),
            spear_corr = spearmanr(x,y)
        )
        _stats_result = {**dict(zip(grouppers, dfn)), 'N': len(x)}
        for sname, s in _stats.items():
            _stats_result[f"{sname}_stat"] = s.statistic
            _stats_result[f"{sname}_p"] = s.pvalue
        stats_result.append(_stats_result)
    corr_stats = pd.DataFrame(stats_result)

    sts = []
    for _, row in corr_stats.iterrows():
        t = "|".join(row[grouppers].to_list())
        sts.append(f"{t}: pearson r={row['pear_corr_stat']:.2f}, p={fmt_pval(row['pear_corr_p'])}")
    
    z, p = compare_correlations(*corr_stats['pear_corr_stat'].values, *corr_stats['N'].values)
    sts_summary = "\n".join(sts + [f"Fisher r-to-z test: z={z:.2f}, p={fmt_pval(p)}"])

    return corr_stats, sts_summary

def compare_correlations(r1, r2, n1, n2):
    """ 
    Determine whether 2 correlations are significantly different using
        Fisher r-to-z test (assumes independent correlation coefficients)

        Args: 
            r1, r2: correlation coeffiients
            n1, n2: number of samples 
    """
    from scipy.stats import norm

    # Fisher r-to-z transform
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))

    # standard error
    se = np.sqrt(1/(n1 - 3) + 1/(n2 - 3))

    # z statistic for difference
    z = (z1 - z2) / se

    # two-tailed p-value
    p = 2 * (1 - norm.cdf(abs(z)))

    return z, p


def get_outliers_col(df, col_name):
    """ given dataframe and column name returns outlier values using pingouin.madmedianrule """
    from pingouin import madmedianrule
    ols = madmedianrule(df[col_name])
    cs = df[col_name][ols]
    return cs

