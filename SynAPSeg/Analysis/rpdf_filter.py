
import numpy as np
from tabulate import tabulate
import pandas as pd
import re
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # add parent dir to sys.path
from utils import utils_general as ug

def filter_rpdf(
    threshold_dict,
    rpdf_coloc,
    all_colocal_ids,
    group_labels_col="img_name",
    return_labels=False,
    clc_nucs_info=None,
):
    """
    DESCRIPTION
        main function to filter dataframe of region props
    ARGS
    - threshold_dict (dict) --> dictionary mapping colocal_ids to dict of region_props (col in df) and min/max vals
        e.g.   threshold_dict = {
                    '0': {'intensity_mean': [1, None], 'area': [50, 1000]}, 
                    '1': {'intensity_mean': [200, None], 'area': [None, 500]}
                    '2': {'intensity_mean': [200, None], 'area': [None, 500]}
                }
    - rpdf_coloc (pd.DataFrame) --> df containing region props
    - group_labels_col (str, None) --> col in rpdf_coloc used to separate duplicate nuclei labels when combining rpdfs 
        e.g. from different images
    - return_labels (bool) --> whether to return valid/invalid labels
    - clc_nucs_info (dict) --> dictionary mapping colocal_ids that are result of colocalization to dict of:
        - intersecting_label_columns (str) --> name of column containing intersecting labels
        - intersecting_colocal_id (int) --> colocal_id value of intersecting nuclei
        e.g.    clc_nucs_info={
                    2: {
                        "intersecting_label_columns":    "ch0_intersecting_label",
                        "intersecting_colocal_id":        0,
                    }
                },
    returns:
        thresholded_rpdf, filtered_counts
            or if return_labels is True
        thresholded_rpdf, filtered_counts, valid_labels, invalid_labels

    """
    # TODO !!! - not sure this respects downgrading colocal ids if intersection percent is lower than threshold

    # format threshold dict - e.g. if using shorthand notations like setting thresholds for all clc_ids
    threshold_dict = format_threshold_dict(threshold_dict, all_colocal_ids)

    # validate input
    verify_threshold_params(threshold_dict, rpdf_coloc)

    # prepare ouputs
    thresholded_rpdfs, valid_labels, invalid_labels, filtered_counts = (
        {},
        {},
        {},
        {"init": get_colocal_id_counts(rpdf_coloc, all_colocal_ids), "final": None},
    )
    # prepare filter filtered_counts for each colocal_id
    for c in all_colocal_ids:
        filtered_counts[c] = {}
    # get present colocal_ids
    present_colocal_ids = sorted(rpdf_coloc["colocal_id"].unique())
    # assign cols to df to track labels by img_name
    rpdf_coloc, new_cols, label_col, clc_nucs_info = prepare_group_label_columns(
        rpdf_coloc, group_labels_col, clc_nucs_info
    )

    
    # MAIN CLC LOOP FOR FILTERING
    #############################
    for colocal_id in all_colocal_ids:
        cdf = rpdf_coloc.loc[rpdf_coloc["colocal_id"] == colocal_id]
        all_labels = set(cdf[label_col].to_list())

        # threshold regionprops
        if colocal_id in threshold_dict:
            cdf, rp_filtercounts = region_prop_table_filter(cdf, threshold_dict[colocal_id])
            for k, v in rp_filtercounts.items():
                filtered_counts[colocal_id][k] = v

        # cdf = clean_up_colocal_nuclei(cdf, colocal_id, clc_nucs_info, valid_labels, group_labels_col)
        if (clc_nucs_info is not None) and (colocal_id in clc_nucs_info):
            # this was reworked to suport multiple intersecting labels
            # get col name with intersecting label and intersecting colocal id
            clc_info = clc_nucs_info[colocal_id]
            for intersecting_label_column, intersecting_colocal_id in zip(
                clc_info["intersecting_label_columns"],
                clc_info["intersecting_colocal_ids"],
            ):
                drp_prev_col = (
                    intersecting_label_column
                    if "grouped_col" not in clc_info
                    else clc_info["grouped_col"]
                )

                # remove  nuclei that are colocal with same object
                #``````````````````````````````````````````````````
                cdf = drop_duplicate_colocalizations(
                    cdf, intersecting_label_column, group_by_col=group_labels_col
                )
                filtered_counts[colocal_id][f"post_drop_duplicates_ch{intersecting_colocal_id}"] = \
                    get_colocal_id_counts(cdf, [colocal_id])[colocal_id]

                # remove any colocal nuclei where overlapping zif nuc was removed previously (e.g. area/axis length)
                cdf = drop_previously_removed(
                    cdf, drp_prev_col, valid_labels[intersecting_colocal_id])
                
                filtered_counts[colocal_id][
                    f"post_drop_previously_removed_ch{intersecting_colocal_id}"
                ] = get_colocal_id_counts(cdf, [colocal_id])[colocal_id]

        valid_labels[colocal_id] = cdf[label_col].to_list()
        invalid_labels[colocal_id] = list(all_labels - set(valid_labels[colocal_id]))
        thresholded_rpdfs[colocal_id] = cdf

    thresholded_rpdf = pd.concat(thresholded_rpdfs.values())
    filtered_counts["final"] = get_colocal_id_counts(thresholded_rpdf, all_colocal_ids)
    if return_labels:
        return thresholded_rpdf, filtered_counts, valid_labels, invalid_labels
    return thresholded_rpdf, filtered_counts


def region_prop_table_filter(cdf, t_dict, inclusive=False) -> tuple[pd.DataFrame, dict[str, int]]:
    ''' helper function to filter region props dataframe using dictionary of min,max vals 
        
        args:
            t_dict: 
                e.g. {'intensity_mean': [1, None], 'area': [50, 1000]}
                
            inclusive (bool), default=False. if true include min/max value in range, else exlcude. 
                e.g. inclusive=True -->  vals >= minval, inclusive=False --> vals > minval
        
        returns:
            cdf, filter_counts_post_thresh
    '''
    filter_counts_post_thresh = {"pre_filters": len(cdf)}
    t_dict = {k: _sanitize_range(v[0], v[1]) for k,v in t_dict.items()}

    for prop_name, (min_value, max_value) in t_dict.items():
        if inclusive: # change behavior to be inclusive of edge values 
            cdf = cdf.loc[(cdf[prop_name] >= min_value) & (cdf[prop_name] <= max_value)]
        else:
            cdf = cdf.loc[(cdf[prop_name] > min_value) & (cdf[prop_name] < max_value)] 
        filter_counts_post_thresh[f"post_{prop_name}"] = len(cdf)

    return cdf, filter_counts_post_thresh

def _sanitize_range(min_value, max_value) -> tuple[float, float]:
    """ convert none in min or max values to -inf or inf """
    min_limit = float("-inf") if min_value is None else min_value
    max_limit = float("inf") if max_value is None else max_value
    return min_limit, max_limit

def pretty_print_fcounts(filtered_counts, title=""):
    """print counts after each filtering step as a str formatted table"""
    count_tabulate = {
        k: v for k, v in filtered_counts.items() if k not in ["init", "final"]
    }
    df_idx = list(
        count_tabulate[
            list(count_tabulate.keys())[
                np.argmax(np.array([len(d) for d in count_tabulate.values()]))
            ]
        ].keys()
    )
    count_f_df = pd.DataFrame.from_dict(count_tabulate).reindex(df_idx)
    print(f"\n{title}\n", tabulate(count_f_df.fillna(""), headers="keys"), flush=True)


def drop_duplicate_colocalizations(
    rpdf_colocal, intersecting_label_column, group_by_col="img_name", as_index=False, 
    metric = "intersection_percent", keep_largest_value = True,
):
    """
    drop colocal detections that are colocal with same object by selecting one with the highest/lowest metric.
    default is keeping highest intersection_percent
    """
    if group_by_col is None:
        dropped_dupes = rpdf_colocal.sort_values(
            [intersecting_label_column, metric], ascending=[True, not keep_largest_value]
        ).drop_duplicates(subset=[intersecting_label_column], keep="first")
    else:
        if not as_index:  # default
            dropped_dupes = rpdf_colocal.groupby(group_by_col, as_index=as_index).apply(
                lambda x: x.sort_values(
                    by=[intersecting_label_column, metric],
                    ascending=[True, not keep_largest_value],
                ).drop_duplicates(subset=[intersecting_label_column], keep="first")
            )
        else:
            dropped_dupes = rpdf_colocal.groupby(group_by_col).apply(
                lambda x: x.sort_values(
                    by=[intersecting_label_column, metric],
                    ascending=[True, not keep_largest_value],
                ).drop_duplicates(subset=[intersecting_label_column], keep="first")
            )
    try:
        dropped_dupes = (
            dropped_dupes.droplevel(level=0)
            if group_by_col is not None
            else dropped_dupes
        )
    except (ValueError, AttributeError):
        pass
    return dropped_dupes


def prepare_group_label_columns(rpdf_coloc, group_labels_cols, clc_nucs_info):
    """assign col to df to track labels by img_name"""
    itx_lbl_col = "intersecting_label_columns"
    og_cols = set(rpdf_coloc.columns.to_list())
    if group_labels_cols is not None:
        rpdf_coloc = rpdf_coloc.assign(
            grouped_labels=rpdf_coloc[group_labels_cols].astype("str")
            + "###"
            + rpdf_coloc["label"].astype("str")
        )
        if clc_nucs_info is not None:
            for k, v in clc_nucs_info.items():
                grp_col_name = f"grouped_labels_{v[itx_lbl_col]}"
                clc_nucs_info[k]["grouped_col"] = (
                    grp_col_name  # add new col name to clc_nuc_info
                )
                # rpdf_coloc[grp_col_name] = np.nan
                rpdf_coloc.loc[
                    ~rpdf_coloc[v[itx_lbl_col]].isna(), grp_col_name
                ] = (
                    rpdf_coloc[group_labels_cols].astype("str")
                    + "###"
                    + rpdf_coloc[v[itx_lbl_col]]
                    .fillna(-1)
                    .astype("int")
                    .astype("str")
                )
    else:
        # if group_labels_col not set, check if need to group labels, count duplicate labels for each colocal_id
        assert (
            rpdf_coloc.groupby("colocal_id")
            .apply(lambda group: group["label"].duplicated().sum())
            .sum()
            == 0
        ), "there are duplicate labels for each colocal_id, set group_labels_col to separate labels by image they came from"
    new_cols = list(set(rpdf_coloc.columns.to_list()) - og_cols)
    label_col = "label" if len(new_cols) == 0 else new_cols[0]
    return rpdf_coloc, new_cols, label_col, clc_nucs_info


def drop_previously_removed(rpdf_colocal, intersecting_label_column, keep_labels_list):
    """remove any colocalized detections where overlapping label was removed previously (e.g. area/axis length)"""
    return rpdf_colocal[rpdf_colocal[intersecting_label_column].isin(keep_labels_list)]


def get_colocal_id_counts(rpdf, all_colocal_ids=None):
    """
    all_colocal_ids (list): sorted list of every colocal id included in ImgDB
        most of the time need to include all ids so 0 counts are recorded, but leave it optional for some usecases
    
    """
    all_colocal_ids = sorted(rpdf['colocal_id'].unique()) if all_colocal_ids is None else all_colocal_ids
    vc = rpdf.value_counts("colocal_id").to_dict()
    return {i: 0 if i not in vc else vc[i] for i in all_colocal_ids}


def format_threshold_dict(threshold_dict, all_colocal_ids):
    """ build threshold dict if using shorthand notations to apply globally
        
        example usage:
        ``````````````
            format_threshold_dict(
                threshold_dict = {
                    'area': [50, 100],
                    'intensity_mean': [200, 2000],
                    '0': {'area': [20, 200]},
                    '1': {'intensity_mean': [0, None]}
                },
                all_colocal_ids = ['0', '1', '2']
            )

            would return: {
                0: {'area': [20, 200], 'intensity_mean': [200, 2000]},
                1: {'area': [50, 100], 'intensity_mean': [0, None]}
                2: {'area': [50, 100], 'intensity_mean': [200, 2000]}
            }
    """
    ftd = {} # formatted threshold dict
    clc_ids = [str(cid) for cid in all_colocal_ids] # convert to strings
    threshold_dict = {str(k):v for k,v in threshold_dict.items()} # format all keys as strings
    
    # separate global from clc_id specifc threshold props, assume if key is not in clc_ids it is a global threshold
    global_props = [k for k in threshold_dict if k not in clc_ids]
    clc_threshes = [k for k in threshold_dict if k not in global_props] # list of clc ids that are manually specified
    
    if not global_props:
        return threshold_dict
    
    # first set all global props
    for prop in global_props:
        for cid in clc_ids:
            if cid not in ftd:
                ftd[cid] = {}
            val_range = threshold_dict[prop]
            ftd[cid][prop] = val_range
    
    # iterate over clc_id specific props and override value range
    for clc_id in clc_threshes:
        clc_props = threshold_dict[clc_id]
        for prop, val_range in clc_props.items():
            ftd[clc_id][prop] = val_range
    
    # change all keys to ints
    ftd = {int(k):v for k,v in ftd.items()}
    return ftd


def verify_threshold_params(threshold_dict, rpdf):
    ignore_missing_regex = "ch\d+_intensity"
    errors = {}
    for clc_id, prop_dict in threshold_dict.items():
        for prop, values in prop_dict.items():
            if prop not in rpdf.columns:
                if re.match(
                    ignore_missing_regex, prop
                ):  # ignore errors for these cols that are calculated while thresholding
                    continue
                if clc_id not in errors:
                    errors[clc_id] = []
                errors[clc_id].append(prop)
    if len(errors) > 0:
        raise KeyError(
            print(
                f"threshold props not found in rpdf:\n\t{errors}\n\trpdf cols: {rpdf.columns.to_list()}"
            )
        )
    return 0

def example_imgdb():
    """ generate an example ImgDB object """
    from utils import utils_ImgDB
    return utils_ImgDB.ImgDB(
        image_channels = [
            {'name': 'MAP2', 'ch_idx': 0, 'colocal_id': 0}, 
            {'name': 'Traf3-eGFP', 'ch_idx': 1, 'colocal_id': 1},
            {'name': 'Psd95', 'ch_idx': 2, 'colocal_id': 2},
        ],
        colocal_nuclei_info = [
                {'name': 'Traf3-eGFP+Psd95', 'ch_idx': [1, 2], 'co_ids': [1, 2], 'colocal_id': 3}, 
                {'name': 'MAP2+Traf3-eGFP+Psd95', 'ch_idx': [0, 1, 2], 'co_ids': [0, 1, 2], 'colocal_id': 4}
            ]
    )


def example_rpdf(imgdb=None, clc_id_p=None, seed_value=42, num_rows = 10000, exclude_copy_vals=['colocal_id', 'label']):
    """ generate a demo rpdf (region prop dataframe) with colocal_ids 0,1,2 
        where colocal_id 2 represents 1 colocalized with 0
    """
    imgdb = imgdb if imgdb is not None else example_imgdb() # not guarenteed to work for all cases, use example if fails
    clc_id_p = [.20, .35, .30, .10, .05] if clc_id_p is None else clc_id_p # must be same len as imgdb.colocal_ids, sum to 1, and must be less than p for base ids
    np.random.seed(seed_value)

    # Create base dataframe
    df = pd.DataFrame({
        'colocal_id': np.random.choice(list(imgdb.colocal_ids.keys()), size=num_rows, p=clc_id_p),  # Random selection of colocal_id
        'label': np.nan,
        'area': np.random.randint(1, 1001, size=num_rows),  # Area values between 1 and 1000
        'intensity_mean': np.random.randint(0, 2**16, size=num_rows),  # Intensity values up to 65535
        'intersection_percent': np.nan,
    })

    # add cols to store intersecting labels for colocalizations
    for col in sorted(set(ug.flatten_list([clz['intersecting_label_columns'] for clz in imgdb.colocalizations]))):
        df[col] = np.nan

    
    # set labels so clc_id 3 uses clc 1 as base and intersects with clc 2, etc..
    for cid, clc_info in imgdb.colocal_ids.items():
        
        # generate labels for this colocal_id
        inds = df['colocal_id']==cid
        n_lbls = sum(inds)
        lbls = np.arange(1, n_lbls+1)
        
        if cid not in imgdb.colocalized_ids:
            df.loc[inds, 'label'] = lbls
            continue
        
        # if colocalization need to set other relevant column values
        clz_info = imgdb.get_colocalization_info(cid)
        base_id = clz_info['base_colocal_id']

        # valid labels are only those that are contained within all inherited_ids
        inherited_clc_ids = imgdb.get_inherited_colocalizations(cid)
        label_lists = [list(df[df['colocal_id']==i]['label']) for i in inherited_clc_ids]
        valid_lbls = ug.intersection_of_lists(*label_lists)
        lbls = sorted(np.random.choice(list(valid_lbls), size=n_lbls, replace=False))
        df.loc[inds, 'label'] = lbls

        # set e.g. area and intensity_mean to be values given by lbl
        base_df = df[(df['colocal_id']==base_id) & (df['label'].isin(lbls))].sort_values('label')
        copy_col_vals = [c for c in base_df.columns if c not in exclude_copy_vals]
        df.loc[inds, copy_col_vals] = base_df.loc[:, copy_col_vals].values
        df.loc[inds, 'intersection_percent'] = np.random.random(n_lbls)

        # set intersecting labels so that they validly point to same itx labels as any of the inherited ids
        #########################################################
        # if only 1 itx lbl it is simple since can be any of them
        if len(clz_info['intersecting_label_columns']) == 1:
            itx_cid = clz_info['intersecting_colocal_ids'][0]
            itx_lbls = list(df[df['colocal_id']==itx_cid]['label'])
            df.loc[inds, clz_info['intersecting_label_columns']] = np.random.choice(itx_lbls, size=n_lbls, replace=False)
        
        else:
            # if more than one need to use same itx label for any shared columns
            # note this isn't tested to handle multiple levels of inheritance e.g. if clc 6 inherited both 5 and 4
            inherited_non_base_ids = [el for el in inherited_clc_ids if el != clz_info['base_colocal_id']]
            for i in inherited_non_base_ids:
                i_info = imgdb.get_colocalization_info(i)
                
                # handle shared itx columns
                shared_itx_cols = list(set(clz_info['intersecting_label_columns']).intersection(set(i_info["intersecting_label_columns"])))
                i_df = df[(df['colocal_id']==i) & (df['label'].isin(lbls))].sort_values('label')
                for slc in shared_itx_cols:
                    df.loc[inds, slc] = i_df[slc].values
                
                # handle non shared itx columns
                non_shared_itx_cols = [c for c in clz_info['intersecting_label_columns'] if c not in shared_itx_cols]
                for nsic in non_shared_itx_cols:
                    itx_cid = clz_info['intersecting_colocal_ids'][clz_info['intersecting_label_columns'].index(nsic)]
                    itx_lbls = list(df[df['colocal_id']==itx_cid]['label'])
                    df.loc[inds, nsic] = np.random.choice(itx_lbls, size=n_lbls, replace=False)
    return df



if __name__ == '__main__':
    
    imgdb = example_imgdb()
    rpdf = example_rpdf(seed_value=42)

    threshold_dict = {
        'area': [50, 1000],
        'intensity_mean': [200, None],
        '0': {'area': [20, 200]},
        '1': {'intensity_mean': [0, None]}
    }
    all_colocal_ids = list(imgdb.colocal_ids.keys())
    clc_nucs_info = imgdb.get_clc_nuc_info()

    f_rpdf, filtered_counts = filter_rpdf(
        threshold_dict, rpdf, all_colocal_ids, 
        group_labels_col=None, return_labels=False, 
        clc_nucs_info=clc_nucs_info
    )


    