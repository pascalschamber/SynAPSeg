import os
import sys
from matplotlib.image import thumbnail
import pandas as pd
import logging
import traceback
from pathlib import Path
import matplotlib.pyplot as plt
from pyparsing import line
import seaborn as sns
import ast
from matplotlib.text import Text, TextPath
from matplotlib.patches import Patch
import scipy.stats
from scipy.stats import mannwhitneyu
import numpy as np
import pingouin as pg

from SynAPSeg.utils import utils_general as ug
from SynAPSeg.utils import utils_plotting as up
from SynAPSeg.utils import utils_image_processing as uip
from SynAPSeg.utils import utils_colocalization as uc
from SynAPSeg.common.Logging import get_logger
from SynAPSeg.config import constants 
from SynAPSeg.IO.project import Project, Example
from SynAPSeg.IO.env import verify_and_set_env_dirs
from SynAPSeg.utils import utils_stats
from SynAPSeg.Analysis import df_utils
from SynAPSeg.Plugins.ABBA import ABBAv2_core_compile as Compile
import SynAPSeg.Analysis.NeighborhoodAnalysis as NNA







####################################################################################
# setup
####################################################################################
# setup global figure settings
up.set_global_textsize_defaults(10)
OUTPUT_DIRNAME = 'outputs'
FIG_FILETYPE = "svg"
outputs_folder = None


#########
# Config
####################################################################################
# optionally set custom env vars - note: os.environ['PROJECTS_ROOT_DIR'] is required to load the current project

verify_and_set_env_dirs({
    # 'PROJECTS_ROOT_DIR': r"J:\SEGMENTATION_DATASETS",
})
PROJECT_NAME = None
outputs_folder = r"D:\OneDrive - Tufts\2026_0206_185211_Quantification_20260109_traf3KO_vGatCre_PSD95GFP_PV"
RPDFS_3D = bool(1)

#####################################################################
# load data
####################################################################################
# get project output
if PROJECT_NAME is not None:
    PROJECT_DIR = os.path.join(os.environ['PROJECTS_ROOT_DIR'], PROJECT_NAME)
    project = Project(PROJECT_DIR)
    OUTPUTS_BASE_DIR = os.path.join(PROJECT_DIR, OUTPUT_DIRNAME)
    output_dir = outputs_folder or ug.get_contents(OUTPUTS_BASE_DIR)[-1] # TODO need to get most recent by default
    output_dir = os.path.join(OUTPUTS_BASE_DIR, output_dir)

    # get project info
    #########################################################################
    ex_i = 1
    ex = project.examples[ex_i]
    imgdb = ex.get_clcdb()

    # TEMP WORK AROUBD before need to fix metadata
    scaling_str = ex.exmd['image_metadata']['scaling']

else:  # i.e. running directly on an output folder (not from project)
    assert outputs_folder is not None, 'outputs_folder must be specified when PROJECT_NAME is None'
    PROJECT_DIR = None
    output_dir = outputs_folder
    OUTPUTS_BASE_DIR = outputs_folder
    
    # get info from quant config copy
    quant_config_path = ug.get_contents(output_dir, filetype='QUANT_CONFIG.yaml')[0]

    from SynAPSeg.IO.BaseConfig import read_config
    quant_config = read_config(quant_config_path, raw=False)

    scaling_str = quant_config['dispatcher_configs'][0]['exmd']['image_metadata']['scaling']

    
FIGURE_OUTDIR = ug.verify_outputdir(os.path.join(
    output_dir, 
    # r"C:\Users\pscham01\OneDrive - Tufts\2025_1212_044118_Quantification_2025_0928_hpc_psd95andrbPV_zstacks",
    f'figures_{ug.get_datetime()}'))

SAVE = bool(1)

# set groupping and feature cols
#########################################################################

BASIC_GROUPPING_COLS = [
    'colocal_id', 'treatment',
    'anid', 'batch', 
]
ALL_FEATURE_COLS = [
    'intensity_mean','intensity_std', 'intensity_min', 'intensity_max', 'integrated_intensity', 'intensity_normByMean',
    'num_pixels', 'area', 'area_um3', 'axis_major_length', 
    'solidity',  'skewedness', 'kurtosis', 
    'roi_distance3D',
]

# parse scaling info
scaling = ast.literal_eval(scaling_str)
scaling = {k:v*1E6 for k,v in scaling.items()}
scaling_factor = np.prod([scaling[dim] for dim in ('ZYX' if RPDFS_3D else 'YX')])

# get colocal id info
imgdb_dict = read_config(os.path.join(output_dir, 'colocal_ids.yaml'))
ALL_COLOCAL_IDS = list(imgdb_dict['colocal_ids'].keys())
HAS_COLOCALIZATIONS = bool(len(imgdb_dict['colocalizations']))
base_colocal_id_map = {d['assign_colocal_id']:d['base_colocal_id'] for d in imgdb_dict['colocalizations']}
ch2clc_map = imgdb_dict['colocalid_ch_map']
clc2itx_map = {d['assign_colocal_id']:d['intersecting_colocal_ids'] for d in imgdb_dict['colocalizations']}

# get data
DATA = {'all_summaries':[], 'all_roi_dfs':[], 'all_rpdfs':[]}
DATA = {Path(p).stem: pd.read_csv(p) for p in ug.get_contents(output_dir, filetype='csv')}

####################################################################################
# copy data from source
####################################################################################
sumdf = DATA['all_summaries'].copy(deep=True)
roidf = DATA['all_roi_dfs'].copy(deep=True)
rpdfs = DATA['all_rpdfs'].copy(deep=True).drop(columns=['coords'])

# general data to extract
####################################################################################
sumdf['size'] = sumdf['area'] * scaling_factor
sumdf['roi_longest_skeleton_path'] = sumdf['roi_longest_skeleton_path'] * scaling['X']
sumdf['count_per_dist'] = sumdf['count'] / sumdf['roi_longest_skeleton_path']

# extra stats + groups
####################################################################################
rpdfs['integrated_intensity'] = rpdfs['area'] * rpdfs['intensity_mean']

# PROJECT SPECIFIC PARAMS 
####################################################################################
# add pv dends intensity
pv_dends_intensity = sumdf[sumdf['colocal_id'] == 1][['ex_i', 'roi_i', 'roi_intensity_mean']]
sumdf = sumdf.merge(pv_dends_intensity, on=['ex_i', 'roi_i'], suffixes=('', '_PV'))


# filter data
####################################################################################
ALL_COLOCAL_IDS = [0]
sumdf = sumdf[sumdf['colocal_id'].isin(ALL_COLOCAL_IDS)]
roidf = roidf[roidf['colocal_id'].isin(ALL_COLOCAL_IDS)]
rpdfs = rpdfs[rpdfs['colocal_id'].isin(ALL_COLOCAL_IDS)]


# colocalization thresholding - default params
ALL_COLOCAL_IDS = sorted(rpdfs['colocal_id'].unique())
THRESHOLD = 0.01
thresholds_by_colocal = dict(zip(ALL_COLOCAL_IDS, [{'area':(0,np.inf)}] * len(ALL_COLOCAL_IDS)))




# plot params 
####################################################################################################

DROP_COLS = [
    'ex_i', 
    'roi_i',
    'scene_name', 
    'label',
    'roi_distance3D',
    'roi_i_byDistance3D',
    'roi_i_byOverlap3D',
    'roi_polyi_byDistance3D',
    'roi_polyi_byOverlap3D',

]
PROP_COLS = [ # a.k.a. region properties and the like
    'region_area_mm',
    'count',
    'density',
    'area',
    'solidity',
    'intensity_std',
    'num_pixels',
    'intensity_mean',
    'axis_major_length',
    'perimeter',
    'axis_minor_length',
    'intensity_max',
    'intensity_min',
    'eccentricity',
    'skewedness', 
    'kurtosis', 
    'circularity',
]

treatment_col = 'treatment'
PALETTE_TREATMENT = {'CON':'gray', 'KO':'red'}
# PALETTE_ANID = up.palette_cat(sumdf, 'anid')
# PALETTE_BATCH = up.palette_cat(sumdf, 'batch')
sp_palette = {k:'k' for k in PALETTE_TREATMENT.keys()}
relabel_keys = {'CON':'CON', 'KO':'KO'}
PALETTE_HUE = PALETTE_LEG = PALETTE_TREATMENT
PALETTE_HUE_LIGHT = PALETTE_TREATMENT
PALETTE_HUE_DARK = PALETTE_TREATMENT
SP_PALETTE_HUE = {k:'k' for k in PALETTE_HUE.keys()}


# PALETTE_STYLES = {'sex':{'M': 'o', 'F': 'X'}}
# STYLE_LABELS={'sex':{'M':'Male', 'F':'Female'}}

FV_LABELS = {
    'count_per_um': 'density (count/$\\mu\\mathrm{m}^3$)' if RPDFS_3D else 'density (count/$\\mu\\mathrm{m}^2$)',
    'count_per_dist': 'density (count/$\\mu\\mathrm{m}$)',
    'intensity_mean': 'mean intensity (A.U.)',
    'area_um3':'puncta size ($\\mu\\mathrm{m}^3$)',
    'area': 'puncta size (pixels)',
    'size': 'puncta size ($\\mu\\mathrm{m}^3$)' if RPDFS_3D else 'puncta size ($\\mu\\mathrm{m}^2$)',
    'count': 'count (total sum)',
}

def FYL(x):
    """ return the formated column label if in FV_LABELS, else return x """
    return FV_LABELS.get(x, x)

# add channel names
####################################################################################
clc2ch_map = {k: d['name'] for k,d in imgdb_dict['colocal_ids'].items()}
roi2name_map = {} #{0:'background', 1:'dendrite', 2:'soma'}
sumdf = sumdf.assign(
    channel_name=lambda x: x['colocal_id'].map(clc2ch_map),
    roi_name=lambda x: x['roi_i'].map(roi2name_map),
)
rpdfs = rpdfs.assign(
    channel_name=lambda x: x['colocal_id'].map(clc2ch_map),
    roi_name=lambda x: x['roi_i'].map(roi2name_map),
)

# filter data
####################################################################################

# aggregate data
####################################################################################
andf = sumdf.groupby(['treatment', 'sex', 'animal_id', 'channel_name', 'colocal_id']).agg({
    'size':'mean',
    'intensity_mean':'mean',
    'count_per_um':'mean',
    'count_per_dist':'mean',
    'count':'sum',
    'roi_area_um': 'sum',
    'roi_longest_skeleton_path': 'sum',
    'roi_intensity_mean': 'mean',
    'roi_intensity_mean_PV':'mean',
}).reset_index()

denddf = sumdf.groupby(['treatment', 'sex', 'ex_i', 'roi_i', 'channel_name', 'colocal_id']).agg({
    'size':'mean',
    'intensity_mean':'mean',
    'count_per_um':'mean',
    'count_per_dist':'mean',
    'count':'sum',
    'roi_area_um': 'mean',
    'roi_longest_skeleton_path': 'mean',
    'roi_intensity_mean': 'mean',
    'roi_intensity_mean_PV':'mean',
}).reset_index()

andf.to_csv(
    os.path.join(FIGURE_OUTDIR, ug.clean_path_name(
        f"animal_aggregated_data_{ug.get_datetime()}.csv")
    ))
denddf.to_csv(
    os.path.join(FIGURE_OUTDIR, ug.clean_path_name(
        f"dendrite_aggregated_data_{ug.get_datetime()}.csv")
    ))

from df_utils import query_df
# basic plots
####################################################################################################
get_rois = [1,2]
get_clcs = [4,5]

_plt_props = ['size', 'intensity_mean', 'count_per_dist', 'roi_intensity_mean_PV', 'roi_area_um', 'roi_longest_skeleton_path']
figures = {
    # 'ePSDs (Bassoon+PSD95-X2)': {'data': sumdf.query("(roi_i.isin(@get_rois)) & (colocal_id.isin([4]))")},
    # 'iPSDs (Gad67+Bassoon)': {'data': sumdf.query("(roi_i.isin(@get_rois)) & (colocal_id.isin([5]))")}
    'PSD95 animal props': {'data': andf},
    'PSD95 dendrite props': {'data': denddf},
}

X = 'treatment'
HUE = 'treatment'
STYLE = None
ORDER = None
STATS_TITLE = bool(1)

for figname, figkwargs in figures.items():
    pltdf = figkwargs['data']
    # auto config
    if ORDER is None:
        ORDER = pltdf[X].unique()

    fig, axs, axis_iterator = up.subplots(len(_plt_props), n_rows=1, size_per_dim=(2, 3))
    fig.suptitle(figname)
    for i, ax in axis_iterator:
        Y = _plt_props[i]
        common_kwargs = dict(
            data=pltdf,
            x = X, y = Y, ax=ax, 
            legend=False,
            hue = HUE,
            hue_order=list(PALETTE_HUE.keys()),
            dodge=bool(0), 
        )
        # sns.boxplot(
        #     ** ug.merge_dicts(common_kwargs, dict(
        #     palette=PALETTE_HUE, 
        #     showfliers=False,
        #     order=ORDER,
        #     zorder=0,
        #     fill=False,
        #     # legend=True if i==0 else False,
        # )))
        sns.barplot(
            ** ug.merge_dicts(common_kwargs, dict(
            palette=PALETTE_HUE, 
            order=ORDER,
            zorder=0,
            fill=False,
            # legend=True if i==0 else False,
        )))
        for sex in ['Male', 'Female']:
            marker = 'x' if sex == 'Female' else 'o'
            subdf = query_df(common_kwargs['data'], **{'sex': sex})
            _palette = {'Male':'k', 'Female':'k'}

            sns.stripplot(
                **ug.merge_dicts(common_kwargs, dict(
                marker=marker,
                # order=_palette.keys(),
                hue_order = _palette.keys(),
                jitter=True,
                zorder=9999,
                size=4,
                linewidth=1,
                edgecolor='k',
                alpha=0.4,
                palette=_palette,
                data=subdf,
                dodge=bool(1),
                hue='sex',
                )))
        ttest_stats = utils_stats.check_ttest_assumptions_and_run(
            pltdf, 
            Y, 'treatment', 'CON', 'KO', plots=False
        )
        if STATS_TITLE:
            ax.set_title(utils_stats.get_ttest_stats_summary(ttest_stats).replace(': ', ':\n'))
        ax.set_ylabel(FYL(Y))
        ax.set_xticks(np.arange(len(ORDER)), ORDER, rotation=30, ha='center')
        ax.set_xlabel('')

    # up.legend_huestyle(ax, HUE, PALETTE_HUE, STYLE, PALETTE_STYLES[STYLE], STYLE_LABELS[STYLE])

    sns.despine(fig)
    plt.tight_layout()

    if SAVE and bool(1): 
        up.save_fig(
            os.path.join(FIGURE_OUTDIR, ug.clean_path_name(
                f"props_PointPlots_{figname}_{ug.get_datetime()}.png")
            ))
    plt.show()


# rpdfs plots 
####################################################################################
ecdf_figures = {
    # 'ePSDs (Bassoon+PSD95-X2)': {'data': rpdfs.query("(roi_i.isin(@get_rois)) & (colocal_id==4)")},
    # 'iPSDs (Gad67+Bassoon)': {'data': rpdfs.query("(roi_i.isin(@get_rois)) & (colocal_id==5)")}
    'PSD95 props': {'data': rpdfs.query("colocal_id==0")},
}
value_cols =  ['intensity_mean', 'area', 'integrated_intensity']

for figname, figkwargs in ecdf_figures.items():
    pltdf = figkwargs['data']
    pltdf = pltdf.copy().melt(id_vars=['treatment'], value_vars=value_cols)
    
    sns.displot(
        pltdf, x='value', hue='treatment', kind='ecdf', col='variable',
        palette=PALETTE_TREATMENT,
        facet_kws=dict(sharex=False),
    )
    plt.suptitle(f"{figname}")
    if SAVE and bool(1): 
        up.save_fig(
            os.path.join(FIGURE_OUTDIR, ug.clean_path_name(
                f"rpdfs_ecdfPlots_{figname}_{ug.get_datetime()}.svg")
            ))
    plt.show()



# build andf by aggregating examples under same animal
####################################################################################################
# - aggregate individual examples by summing counts, recalc densities and re-averaging other value cols
# - above GROUPING_COLS should define 1 or several examples that should be aggregated
val_cols_sum = [] # Base components to sum

# columns that need weighted averaging (all numeric minus the sum-cols and IDs)
val_cols_avg = sumdf.select_dtypes(include=np.number).columns.difference(
    DROP_COLS + GROUPPING_COLS + val_cols_sum + ['density'] 
).to_list()

GROUPPING_COLS_ANDF = list(set(GROUPPING_COLS).difference(DROP_COLS))

andf = (
    sumdf
    .groupby(GROUPPING_COLS_ANDF)
    [val_cols_avg]
    .mean()
    .reset_index()
    .drop(columns=DROP_COLS, errors='ignore')
)




# plot PV intensity correlation w/ density and validation of distributions
####################################################################################################
sns.lmplot(sumdf, x='roi_axis_major_length', y='count', hue='treatment');plt.show()
sns.lmplot(sumdf, x='roi_axis_major_length', y='count_per_um', hue='treatment');plt.show() 
sns.lmplot(sumdf, x='roi_axis_major_length', y='count_per_dist', hue='treatment');plt.show()
sns.lmplot(sumdf, x='roi_axis_major_length', y='count_per_dist', hue='treatment');plt.show()

min_group_size = rpdfs.groupby(['treatment']).size()
print('n samples in rpdfs per treatment group:', min_group_size)

sns.histplot(
    data=rpdfs, 
    common_norm=False,                              # when common_norm=False <-- normalize per hue group
    x='intensity_mean', hue='treatment', 
    stat='percent', element='step', fill=False,
    palette=PALETTE_TREATMENT
); plt.show()

sns.histplot(
    data=rpdfs, 
    common_norm=False,                              # when common_norm=False <-- normalize per hue group
    x='area', hue='treatment', 
    stat='percent', element='step', fill=False,
    palette=PALETTE_TREATMENT
); plt.show()


len(rpdfs)
len(sumdf)
len(sumdf.query("roi_area_um<1000 & roi_area_um>40 & axis_area_ratio<4"))
len(sumdf.query("roi_area_um<1000 & roi_area_um>40"))
# sumdf = sumdf.query("roi_area_um<1000 & roi_area_um>40 & axis_area_ratio<4") # old thresh to help keep only good ones, now not relevant



# plot ecdfs for all dists comp. treatment groups
#################################################
# plot_features = []
# pltdf = [rpdfs, nonPV_rpdfs][0]
# sns.displot(pd.melt(pltdf, ), x='integrated_intensity', hue='treatment', kind='ecdf', palette=PALETTE_TREATMENT)
# plt.show()



    
# corr. PV roi intensity and synapse density
####################################################################################################
_y_vars = ['count_per_um', 'count_per_dist', 'intensity_mean', 'area_um3']
groupers = ['age']
pltdf = sumdf.copy(deep=True)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for _y_var, ax in zip(_y_vars, axes.flatten()):
    for grp in ['3-month', '12-month']:
        sns.regplot(pltdf.query(f"age=='{grp}'"), x='roi_intensity_mean_PV', y=_y_var, color=PALETTE_LEG[grp], ax=ax)
    corr_stats, corr_stats_summary = utils_stats.compare_correlations_stats(pltdf, ['roi_intensity_mean_PV', _y_var], groupers)
    ax.set_title(corr_stats_summary)
    ax.set_ylabel(FYL(_y_var))
plt.tight_layout()
if SAVE: up.save_fig(os.path.join(FIGURE_OUTDIR, f'pvIntensityCorrs_{ug.get_datetime()}.svg'))
plt.show()


# get representative images
####################################################################################
if bool(0): 
    from SynAPSeg.Analysis import df_utils
    import importlib
    importlib.reload(up)
    from SynAPSeg.utils import utils_plotting as up

    rep_animals = df_utils.get_representative_samples(
        sumdf,
        id_vars=["ex_i", "roi_i"], 
        within_vars=['age'],
        value_vars=["count_per_dist", 'roi_intensity_mean_PV'], 
        n=10
    )

    _arr_grid = {
        'n_cols':5, 'cmap':'gray', 'noshow': True, 
        'im_show_kwargs': {'vmin':0, 'vmax':1, 'interpolation':'bilinear'}, 
        'mask_display_function_kwargs':{'imshow_kwargs': {'vmin':0, 'vmax':1, 'interpolation':'bilinear'}},
        'images': [], 'masks':[], 'titles':[]}
    _arr_grid.update(**{'im_show_kwargs': {'vmin':0, 'vmax':1, 'interpolation':'bilinear'}, 
        'mask_display_function_kwargs':{'imshow_kwargs': {'vmin':0, 'vmax':1, 'interpolation':'bilinear'}}})
    for rowi, row in rep_animals.iterrows():
        age = row['age']
        exi = row['ex_i']
        roi_i = row['roi_i']
        example = project.get_example((str(exi)).zfill(4))
        dendimg = uip.read_img(example.get_path('annotated_dends_filt_ch0.tiff'))
        
        sdimg = uip.read_img(example.get_path('pred_stardist_n2v_3d_v310.tiff'))
        pvimg = uip.read_img(example.get_path('raw_img.tiff'))[0,0,1]/(2**16-1)
        psd95img = uip.normalize_01(uip.read_img(example.get_path('pred_n2v2.tiff'))[0,0,1])
        intimg = np.stack([psd95img, pvimg], 0)
        
        extent = uip.find_extent(dendimg==roi_i)
        z0, z1, y0, y1, x0, x1 = extent
        cropped = [v[z0:z1+1, y0:y1+1, x0:x1+1] 
            for v in [dendimg] + uip.unpack_array_axis(sdimg[0,0], 0) + uip.unpack_array_axis(intimg, 0)]
        uip.pai(cropped)
        dendroi = cropped.pop(0)
        dendroi = np.where(dendroi==roi_i, 999999, 0)
        sdpred = cropped.pop(0)
        _ = cropped.pop(0)
        # create lbls img for dend + sd pred for psd95 channel, and dend alone for pv channel
        sdpmip = uip.mip(sdpred)
        psdmask = np.logical_or(uip.mip(dendroi), sdpmip)*999999
        psdmask = np.where(sdpmip>0, sdpmip, psdmask)
        pvmask = uip.mip(dendroi)   
        masks = [psdmask, pvmask]

        _arr_grid['images'].extend([sdpmip] + [uip.mip(c) for c in ([cropped[0]]*2 + [cropped[1]]*2)])
        _arr_grid['masks'].extend([None, None,  masks[0], None] + masks[1:])
        _arr_grid['titles'].extend(
            [f"age: {age}, exi: {exi}, roi_i: {roi_i}, {c}" for c in ['sdpred', 'psd95', 'psd95', 'PV', 'PV']])

    fig, axs = up.plot_image_grid(**_arr_grid)
    for axj in range(axs.shape[0]):
        up.add_scalebar(axs[axj,2], 0.0851, '2 µm', 2, title_y_adjust=10)
        up.add_scalebar(axs[axj,4], 0.0851, '2 µm', 2, title_y_adjust=10)
    up.save_fig(os.path.join(FIGURE_OUTDIR, f'aging_3d_dendrite_analysis_representative_images_{ug.get_datetime()}.svg'))
    plt.show()

    







# roi_plots
####################################################################################
save_roi_plots = bool(1)
roi_sp_pal = {'3-month': 'k', '12-month': 'k'} or PALETTE_BATCH
roipltdf = (
    roidf_pvs
    # roidf
    # result.query("cluster_label==1")
    .groupby(['treatment', 'batch', 'anid', 'colocal_id']).mean(numeric_only=True).reset_index()
    .replace(relabel_keys)
)
# add psd95 intensity
roidf_psd95 = (
    roidf
    .groupby(['treatment', 'batch', 'anid', 'colocal_id']).mean(numeric_only=True).reset_index()
    .replace(relabel_keys)
)
roipltdf = roipltdf.merge(
    (roidf_psd95[['treatment', 'batch', 'anid', 'roi_intensity_mean']]
     .rename(columns={'roi_intensity_mean': 'psd95_roi_intensity_mean'})),
    on = ['treatment', 'batch', 'anid']
)

YVARS = ['roi_area_um', 'roi_axis_major_length', 'roi_intensity_mean', 'psd95_roi_intensity_mean']

ttestDf = utils_stats.merge_mwu_with_ttest_run(
    utils_stats.batch_ttest_run(
        roipltdf[[treatment_col]+YVARS].melt(id_vars=[treatment_col]), 
        y_var='value', iter_col='variable',
        treatment_col=treatment_col, group1='3-month', group2='12-month'
    ),
    utils_stats.batch_mwu_run(roipltdf, YVARS, grp1='3-month', grp2='12-month', treatment_col=treatment_col)
)
utils_stats.summarize_batch_stats(ttestDf)

fig_rows_cols = np.array([1,len(YVARS)])
base_axsize = np.array((1.0, 1.3))*3
fig,axs=plt.subplots(fig_rows_cols[0], fig_rows_cols[1], figsize=base_axsize*fig_rows_cols[::-1])
for i, yvar in enumerate(YVARS):
    ax = axs.flatten()[i]
    sns.boxplot(
        roipltdf, x='treatment', y=yvar, hue='treatment', palette=PALETTE_LEG, fill=False, 
        legend=False, ax=ax, showfliers=False, notch=False, order=PALETTE_LEG.keys(), 
        linewidth=2.5,
    )
    sns.swarmplot(roipltdf, x='treatment', y=yvar, hue='treatment', dodge=False, legend=False, ax=ax, palette=roi_sp_pal, alpha=0.7, order=PALETTE_LEG.keys())
    if i==0:
        up.make_legend(PALETTE_LEG, bbox_to_anchor=(1,1))
plt.tight_layout()
sns.despine(fig)
if save_roi_plots: 
    up.save_fig(os.path.join(FIGURE_OUTDIR, f'roi_analysis_pv.svg'))
    ttestDf.to_csv(os.path.join(FIGURE_OUTDIR, f'roi_analysis_pv_stats.csv'))
plt.show()







####################################################################################
# synaptic neighborhood analysis
####################################################################################
# import importlib
# importlib.reload(NNA)
from SynAPSeg.Analysis import NeighborhoodAnalysis as NNA
import itertools
import scipy.stats
from scipy.stats import ks_2samp
from SynAPSeg.utils import utils_stats as utils_stats
from SynAPSeg.Analysis.plotting import plot_ptile_displot

data = {
    "rpdfs": rpdfs.copy(deep=True)  # your region-props dataframe
}
neighbor_method = ['radius','knn'][1]
radius_list = [1, 1.25, 1.5]
k_list = [1, 2, 4][:1]
config = {
    "scaling": scaling,
    "centroid_col": "centroid",
    "dend_id_col": "dend_id",
    "group_col": "treatment",   # e.g. 'young' / 'old'
    "feature_cols": ["area_um3", 'intensity_mean'][1:],
    "use_z": True,                   # or False for 2D analysis
    "min_neighbors": 1,               # filter out isolated synapses
    "drop_outliers": bool(0),
    "equal_samples": bool(0),
    "method": neighbor_method,
    "radius_list": radius_list,  # same units as centroid coords
    "k_list": k_list,
    "n_permutations": 100,
    "zscore_groupby_cols": ['treatment'],
}

results = NNA.run_synaptic_neighborhood_analysis(data, config)
per_syn = results["per_synapse_stats"]


# validate/review number of neighbors at each radius threshold
print(results['nNeighbors_distribution'])

# run visualization/figures for combinations of features and radius thresholds

SAVE_NN_ANALYSIS = bool(1)
USE_ZSCORE = bool(1) # if false use raw values 

# setup cols to plot
_colname_suffix = '_z' if USE_ZSCORE else '_raw'
FEATURE_VALUE_COL =  f'feature_value{_colname_suffix}'
NEIGHBOR_MEAN_COL =  f'neighbor_mean{_colname_suffix}'
NEIGHBOR_DELTA_COL = f'neighbor_delta{_colname_suffix}'

iter_plot_params = (config['feature_cols'], config['radius_list']) if neighbor_method=='radius' else (config['feature_cols'], config['k_list'])
for feature_col, neighborhood_threshold in itertools.product(*iter_plot_params):
    print("feature_col, neighborhood_threshold", feature_col, neighborhood_threshold)
    nn_col = 'radius' if neighbor_method=='radius' else 'k_neighbors'
    df = per_syn
    mask = (df["feature"] == feature_col) & (df[nn_col] == neighborhood_threshold)
    df_sub = df.loc[mask].copy()
    df_sub['ex_i'] = df_sub['dend_id'].apply(lambda s: int(s.split('_')[0][2:]))

    
    
    
    if config['equal_samples']:
        plt_sample_size = df_sub.groupby(['treatment']).size().min()
        pltdf = df_sub.groupby(['treatment']).sample(plt_sample_size).reset_index()
    else:
        pltdf = df_sub

    pltdf['age'] = pltdf['treatment'].replace(relabel_keys)
    ages = pltdf['age'].unique()
    treatments = pltdf['treatment'].unique()


    plthist_cols = ['neighbor_delta_raw', 'neighbor_delta_z', 'shuffled_neighbor_delta_z']
    fig,axs = plt.subplots(1, len(plthist_cols), figsize=(len(plthist_cols)*4.5, 4), sharey=True)
    for xcol, ax in zip(plthist_cols, axs.flatten()):
        sns.histplot(
            data=df_sub, common_norm=False, ax=ax,
            x=xcol, 
            hue='age', 
            stat='percent', 
            element='step', fill=False,
            palette=PALETTE_LEG,
            legend=False if xcol==plthist_cols[-1] else True,
        )
        stat, p = ks_2samp(
            df_sub.query(f"(age==\'{ages[0]}\')")[xcol].values, 
            df_sub.query(f"(age==\'{ages[1]}\')")[xcol].values
        )
        ax.set_title(f'Relative Frequency Distribution of\n{xcol}({feature_col})\nks_2samp={stat:.2e}, p={p:.2e}')
    if SAVE_NN_ANALYSIS:
        up.save_fig(os.path.join(FIGURE_OUTDIR, f'NN_{NEIGHBOR_DELTA_COL}_overall_displot_f{feature_col}_{neighbor_method}{neighborhood_threshold}.svg'))
    plt.show()

    # plot boxplot on top or bot x percentile
    PTILE = 0.25
    plt_ptile_cols = [
        ('feature_value_z', 'neighbor_delta_z'), 
        ('feature_value_z', 'shuffled_neighbor_delta_z'),
        ('feature_value_raw', 'neighbor_delta_raw'), 
        ('feature_value_raw', 'shuffled_delta_global'), 
        ('feature_value_raw', 'shuffled_delta_local')
    ]
    for _cols in plt_ptile_cols:
        val_col, dist_col = _cols
        ptdf = plot_ptile_displot(
            pltdf, 
            get_ptile_col = val_col, 
            plt_dist_col = dist_col,
            group_cols = ["age"],
            palette=PALETTE_LEG,
            PTILE = PTILE,
            PLOT_MID=True,
            element='step',
            common_norm=False,
            fill=False
        )

        for axi, dfn in enumerate(ptdf['subset'].unique()):
            adf = ptdf[ptdf['subset']==dfn]
            stat, p = ks_2samp(
                adf.query(f"(age==\'{ages[0]}\')")[dist_col].values, 
                adf.query(f"(age==\'{ages[1]}\')")[dist_col].values
            )
            ax = plt.gcf().axes[axi]
            bin_dist_label = dfn.replace(' x.', ' ').replace('`x ','` ')
            ax.set_title(f'Relative Frequency Distribution of\n{bin_dist_label}\nks_2samp={stat:.2e}, p={p:.2e}')
        
        if SAVE_NN_ANALYSIS:
            up.save_fig(os.path.join(FIGURE_OUTDIR, f'NN_{NEIGHBOR_DELTA_COL}_ptile{PTILE}_of{val_col}_distOf{dist_col}_{neighbor_method}{neighborhood_threshold}.svg'))
        plt.show()


    # correlations self vs neighbor
    ###################################################################
    plot_nb_corrs = [
        ('feature_value_z', 'neighbor_mean_z'), 
        ('feature_value_z', 'local_zscore'),
        ('feature_value_raw', 'neighbor_mean_raw'), 
        ('feature_value_raw', 'shuffled_mean_global'), 
        # ('feature_value_raw', 'shuffled_mean_local')
    ]
    CORR_HUE_COL = ['age', 'dend_id', 'ex_i'][0]
    CORR_PALETTE = PALETTE_LEG if CORR_HUE_COL=='age' else None
    for corr_cols in plot_nb_corrs:
        corrx, corry = corr_cols
        corr_g = sns.lmplot(
            # data=df_sub, 
            # col='treatment', 
            # hue='anid',
            data = pltdf,
            x=corrx,
            y=corry,
            scatter = True,
            # alpha=0.1,
            hue=CORR_HUE_COL,
            palette=CORR_PALETTE,
            fit_reg=True,
            scatter_kws = {'s':1, 'alpha':0.25},
            legend=False,
        )
        for ax in corr_g.axes.flatten():
            ax.set_xlabel(f'Self ({corrx})')
            ax.set_ylabel(f'Neighbor ({corry})')
        
        utils_stats.check_ttest_assumptions_and_run(pltdf, corry, treatment_col='age',group1='3-month', group2='12-month', plots=bool(0))

        corr_stats, corr_stats_summary = utils_stats.compare_correlations_stats(pltdf, corr_cols, ['age'])
        ax = corr_g.axes.flatten()[0]
        ax.set_title(corr_stats_summary)

        if SAVE_NN_ANALYSIS and bool(1):
            up.save_fig(os.path.join(FIGURE_OUTDIR, f'NN_{feature_col}_correlation_{corrx}x{corry}by{CORR_HUE_COL}_{neighbor_method}{neighborhood_threshold}.svg'))
            corr_stats.to_csv(os.path.join(FIGURE_OUTDIR, f'NN_{feature_col}_correlation_{corrx}x{corry}by{CORR_HUE_COL}_{neighbor_method}{neighborhood_threshold}.csv'))
        plt.show()

    

    # plot boxplot on top or bot x percentile
    pltdist_cols = ['neighbor_delta_z', 'shuffled_neighbor_delta_z', 'neighbor_delta_raw', 'shuffled_delta_global', 'shuffled_delta_local']
    # calculate min/max across all cols so same x axis for all plots
    # minmax = pltdf[pltdist_cols].quantile([0,1]).values
    # minmax = minmax.min()*1.05, minmax.max()*1.05
    for dist_col in pltdist_cols:
        pTileDf = plot_ptile_displot(
            pltdf, 
            get_ptile_col=FEATURE_VALUE_COL, 
            plt_dist_col = dist_col,
            group_cols = ["age"],
            palette=PALETTE_LEG,
            PTILE = 0.10,
            common_norm=False,
            stat = 'percent', element='step', fill=False,
        )
        for ax in plt.gcf().axes:
            ax.axvline(x=0, color='k', linestyle='--', zorder=0, alpha=0.7)
            # ax.set_xlim(minmax)
        if SAVE_NN_ANALYSIS:
            up.save_fig(os.path.join(FIGURE_OUTDIR, f'NN_{NEIGHBOR_DELTA_COL}_ptile_10th_90th_f{feature_col}_{dist_col}_displot_{neighbor_method}{neighborhood_threshold}.svg'))
        plt.show()
    
    # plot the zscored feature-neighborhood difference
    #################################################################################################
    # upshift refers the percent of synapses that are significantly clustered with bright neighbors
    # this table explains what the zscore means relative to the distribution being plotted
    # e.g. when looking at the the Dimmest 10% (this is flipped for brightest 10%)
    # Positive (Z-Score > 1.96) - Bright Neighbors - Outlier (Low-High) - A dim synapse isolated in a "bright" zone.
    # Negative (Z-Score < -1.96) - Dim Neighbors - Clustered (Low-Low) - A "Coldspot" where many dim synapses clump together.
    
    # INTERP of analysis
    # "When looking at the brightest synapses, is their environment also bright?" 
    # neighbor_mean_z is for magnitude of the similarity
    # local_zscore is for the statistical significance of the similarity

    # implementation questions to address:
    # - does it make more sense to calculate zscore on a per-animal basis?
    # - does it make more sense to calculate zscore on a per-dendrite basis?
    # - currently using per-group zscored but I notice some shifts are due to a couple of animal outliers
    from SynAPSeg.utils import utils_stats
    import importlib
    importlib.reload(utils_stats)
    from scipy.stats import fisher_exact, chi2_contingency

    PTILE = [0.10, 0.25][1]
    continency_stat = 'fisher_exact'

    dist_sig_threshes = {
        'local_zscore': [1.96, -1.96], 
        'neighbor_mean_z': [0.5, -0.5], 
        'neighbor_delta_z': [0.5, -0.5],
        'shuffled_neighbor_delta_z': [0.5, -0.5], 
    }
    plt_zscore_neighbor_shift_cols = ['local_zscore', 'neighbor_mean_z', 'neighbor_delta_z', 'shuffled_neighbor_delta_z']
    figtitles = {
        'local_zscore': 'Is a synapse\'s neighborhood significantly different from the rest of the dendrite?', 
        'neighbor_mean_z': 'Is a synapse similar to it\'s neighborhood?',
        'neighbor_delta_z': 'Is a synapse significantly different from its neighborhood?',
        'shuffled_neighbor_delta_z': 'Is a synapse significantly different from a random neighborhood?',
    }
    all_animals = pltdf['ex_i'].unique() # for ensuring all animals are included in the stats 

    for dist_col in plt_zscore_neighbor_shift_cols:
        zscore_pTileDf = plot_ptile_displot(
            pltdf, 
            get_ptile_col='feature_value_z', 
            plt_dist_col = dist_col,
            group_cols = ["age"],
            palette=PALETTE_LEG,
            PTILE = PTILE,
            common_norm=False,
            stat = 'percent', element='step', fill=False,
        )
        fig = plt.gcf()
        

        sig_threshes = dict(zip(['upshifted', 'downshifted'], dist_sig_threshes[dist_col]))
        
        # # iter over each axis which plots a different percentile bin and add the percent significantly shifted to the title
        zscored_animal_averaged = []
        for ssi, subset in enumerate(zscore_pTileDf['subset'].unique()):
            ax = fig.axes[ssi]
            subset_data = zscore_pTileDf.query(f"subset=='{subset}'")
                        
            new_title = f"{ax.get_title()}\n"
            for shift_name, sig_shift_thresh in sig_threshes.items():
                ax.axvline(x=sig_shift_thresh, color='k', linestyle='--', zorder=0, alpha=0.7)
                if shift_name == 'upshifted':
                    get_p_shifted = lambda x: 0.0 if len(x)==0 else x[x[dist_col] > sig_shift_thresh].shape[0]/x.shape[0]
                else:
                    get_p_shifted = lambda x: 0.0 if len(x)==0 else x[x[dist_col] < sig_shift_thresh].shape[0]/x.shape[0]
                shifted_data = zscore_pTileDf.query(f"subset=='{subset}'").groupby('age').apply(get_p_shifted)
                new_title += f"{shift_name}: " +"|".join([f"{k}:{round(v*100,2)}%" for k,v in shifted_data.to_dict().items()]) 
                
                # determine whether the groups's shift are significantly different
                # Build contingency table for each shift # Row 0: Treatment 1 [sig_count, non_sig_count] # Row 1: Treatment 2 [sig_count, non_sig_count]
                contingency_table = []
                for trt in ages:
                    trt_df = subset_data[subset_data['age'] == trt]
                    if shift_name == 'upshifted':
                        sig_count = (trt_df[dist_col] > sig_shift_thresh).sum()
                    else:
                        sig_count = (trt_df[dist_col] < sig_shift_thresh).sum()
                    non_sig_count = len(trt_df) - sig_count
                    contingency_table.append([sig_count, non_sig_count])
                
                # Run the test
                if continency_stat == 'fisher_exact':
                    stat, p_val = fisher_exact(contingency_table) 
                elif continency_stat == 'chi2': #OR use chi2 if N is very large:
                    stat, p_val, dof, ex = chi2_contingency(contingency_table)
                else:
                    raise ValueError(f"Invalid continency_stat: {continency_stat}")
                
                # 3. Update Title with p-value
                new_title += f"({continency_stat} p={p_val:.4f})\n"

                # add data without taking treatment mean
                shiftdf = (
                    subset_data.groupby(['age','ex_i', 'subset'])
                    .apply(get_p_shifted).reset_index().rename(columns={0:'p_shifted'})
                    .assign(shift=shift_name, continency_stat=continency_stat,statistic=stat, p_val=p_val)
                )
                null_data = []
                for exi in all_animals:
                    if exi not in shiftdf['ex_i'].unique():
                        base = shiftdf.loc[0].copy(deep=True)
                        base['ex_i'] = exi
                        base['p_shifted'] = 0.0
                        base['age'] = pltdf.query(f"ex_i=={exi}")['age'].values[0]
                        shiftdf = pd.concat([shiftdf, base.to_frame().T], ignore_index=True)
                        

                zscored_animal_averaged.append(shiftdf)
            ax.set_title(new_title)
        
        zscored_animal_averaged = pd.concat(zscored_animal_averaged, ignore_index=True)
        statsRes = utils_stats.batch_indSamples_test(
            zscored_animal_averaged, ['p_shifted'], groupby_cols=['subset', 'shift'], treatment_col='age'
        ).reset_index()
        
        if SAVE_NN_ANALYSIS:
            up.save_fig(os.path.join(FIGURE_OUTDIR, f'NN_{dist_col}_clustering_zscores_f{feature_col}_{neighbor_method}{neighborhood_threshold}.svg'))
            zscored_animal_averaged.to_csv(os.path.join(FIGURE_OUTDIR, f'NN_{dist_col}_clustering_zscoresByAnimal_boxplot_f{feature_col}_{neighbor_method}{neighborhood_threshold}.csv'), index=False)
            statsRes.to_csv(os.path.join(FIGURE_OUTDIR, f'NN_{dist_col}_clustering_zscoresByAnimal_boxplot_f{feature_col}_{neighbor_method}{neighborhood_threshold}_stats.csv'), index=False)
        plt.show()

        # plot boxplot of up/down shifts averaged within animals for each subset & group 
        get_test = 'mwu'
        fig,axs = plt.subplots(1, zscored_animal_averaged['subset'].nunique(), sharey=True, figsize=(10,5))
        for i, subset_name in enumerate(zscored_animal_averaged['subset'].unique()):
            zandf = zscored_animal_averaged.query(f"subset=='{subset_name}'")
            zandf_stats = statsRes.query(f"subset=='{subset_name}'")
            ax = axs.flatten()[i]
            sns.boxplot(data=zandf, x='shift', y='p_shifted', hue='age', order=['downshifted','upshifted'], ax=ax, palette=PALETTE_HUE, showfliers=False, fill=False, hue_order=PALETTE_HUE.keys(), legend=i==0)
            sns.swarmplot(data=zandf, x='shift', y='p_shifted', hue='age', order=['downshifted','upshifted'], ax=ax, dodge=True, legend=False, palette=PALETTE_HUE_LIGHT, hue_order=PALETTE_HUE.keys())
            bin_dist_label = subset_name.replace(' x.', ' ').replace('`x ','` ').split(' & ')
            # need to split stats by shift up or down
            ax.set_title("\n".join([f'% significantly shifted `{dist_col}` @'] + bin_dist_label + [
                f"{shiftname}:{_zandf['mwu_statistic'].values[0]:.2f} p={_zandf['mwu_pvalue'].values[0]:.2f}" for shiftname, _zandf in zandf_stats.groupby('shift')
            ]+[f'{get_test}']))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=40)
        fig.suptitle(figtitles.get(dist_col, ''))
        sns.despine(fig)
        plt.tight_layout()
        if SAVE_NN_ANALYSIS:
            up.save_fig(os.path.join(FIGURE_OUTDIR, 
                f'NN_clustering_zscoresByAnimal_boxplot_{feature_col}_ptile{PTILE}_dist{dist_col}_{neighbor_method}{neighborhood_threshold}.svg'))
        plt.show()





if bool(1):    # plot dist for percentile bins - visually similar to seaborn example Overlapping densities (‘ridge plot’) https://seaborn.pydata.org/examples/kde_ridgeplot.html
    ##################################################################################################################################
    
    SAVE_DISPLOT_PERCENTILES = bool(1)
    
    fs = {}
    displot_ptile_bins = list(range(0,91,20))
    displot_ptile_nbins = len(displot_ptile_bins)
    for i in displot_ptile_bins:
        binL, binR = i, i+100//displot_ptile_nbins
        ptile_key = f"{binL}-{binR}"
        fs[ptile_key]=f"lambda x: (x > x.quantile({binL/100 or 0.0001})) & (x < x.quantile({(binR)/100}))"
    
    pTileDf = plot_ptile_displot(
            df_sub, 
            get_ptile_col=FEATURE_VALUE_COL, 
            plt_dist_col = NEIGHBOR_DELTA_COL,
            group_cols = ["treatment"],
            palette=PALETTE_TREATMENT,
            PTILE_FUNCS = fs,
            common_norm=False,
            stat = 'percent', element='step', fill=False,
        )


    sns.set_style('darkgrid')

    # Initialize the FacetGrid object
    g = sns.FacetGrid(
        pTileDf, row="subset", hue="treatment", 
        aspect=4, height=1, 
        palette=PALETTE_TREATMENT, row_order=list(pTileDf['subset'].unique())[::-1]
    )

    # Draw the densities in a few steps
    g.map(sns.kdeplot, NEIGHBOR_DELTA_COL,
        bw_adjust=.5, clip_on=False,
        common_norm=False,
        #   alpha=1, 
        linewidth=1.5, legend=False, fill=True, alpha=0.5,
        # clip=(-2000, 2000)
    )
    g.map(sns.kdeplot, NEIGHBOR_DELTA_COL, clip_on=False, 
        color="w", lw=2, bw_adjust=.5, legend=False, common_norm=False, 
        # clip=(-2000, 2000),
    )

    # Define and use a simple function to label the plot in axes coordinates
    for (plt_idx, _df) in (g.facet_data()):
        subset_val = _df['subset'].unique()[0]
        hue_val = _df['treatment'].unique()[0]
        
        # get the axes
        ax = g.axes.flat[plt_idx[0]]

        # plot the distributions median for each group and subplot
        # _dist_mean = _df[NEIGHBOR_DELTA_COL].mean()
        _dist_mean = np.median(_df[NEIGHBOR_DELTA_COL])
        ax.axvline(_dist_mean, c=PALETTE_TREATMENT[hue_val], ls='--', zorder=9999)
        print(plt_idx, subset_val, hue_val, _dist_mean)     # plt_idx = (row, col, group)

        # annotate each row with subset name
        if plt_idx[2] != 0: # only do this for first hue to avoid dupes
            continue        
        ax.text(0, 0.8, str(subset_val),
                fontweight="bold", color="black",
                ha="left", va="center", transform=ax.transAxes)
        ax.axvline(0,c='k',ls='-',zorder=99999)
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_ylabel("")
        ax.set_ylim(0, None)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color='k', clip_on=False, zorder=99999)

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=0.015)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.despine(bottom=True, left=True)


    from scipy.stats import skewtest, wilcoxon
    
    # run stats for dist plots
    run_tests = {'skew':skewtest, 'wilcoxon':wilcoxon} # D’Agostino’s Skewness Test, Wilcoxon Signed-Rank Test
    NN_dist_stats = []
    for dfn, adf in pTileDf.groupby(['subset']):
        subset_name = dfn[0]
        _grps = adf['treatment'].unique()
        _datas = [adf[adf['treatment']==_grp][NEIGHBOR_DELTA_COL].values for _grp in _grps]

        ks_2samp_stat, ks_2samp_p = ks_2samp(_datas[0], _datas[1])

        _sd = {
            'subset':subset_name,
            'ks_2samp_stat':ks_2samp_stat, 'ks_2samp_p':ks_2samp_p, 
            'g1': _grps[0], 'g2':_grps[1],
        }

        for i, grp in enumerate(_grps):
            for test_name, test_fxn in run_tests.items():
                stat, p = test_fxn(_datas[i])
                _sd[f"g{i+1}_{test_name}_stat"] = stat
                _sd[f"g{i+1}_{test_name}_p"] = p
        NN_dist_stats.append(_sd)
    NN_dist_stats = pd.DataFrame(NN_dist_stats)

    if SAVE_DISPLOT_PERCENTILES:
        up.save_fig(os.path.join(FIGURE_OUTDIR, f'NN_{NEIGHBOR_DELTA_COL}_ptile_bins_f{feature_col}_stacked_displot_{neighbor_method}{neighborhood_threshold}.svg'))
        NN_dist_stats.to_excel(
            os.path.join(FIGURE_OUTDIR, f'NN_{NEIGHBOR_DELTA_COL}_ptile_bins_f{feature_col}_stacked_displot_{neighbor_method}{neighborhood_threshold}_stats.xlsx'), index=False
        )
    plt.show()



####################################################################################
# Per dendrite variability
####################################################################################
# for each ex, get valid dend_ids, get rps on this dend

features = [
    "area", "intensity_mean",
    # "integrated_intensity", "skewedness", "kurtosis", 
    # "axis_major_length", "solidity",  "intensity_normByMean", 
]
grpby_cols = ['treatment', 'batch', 'anid', 'ex_i', 'roi_i']
grpby_cols_reduced = ['treatment', 'batch', 'anid']

# Functions for variability
def compute_variability(df, cols):
    out = {}
    for col in cols:
        sd = df[col].std()
        mean = df[col].mean()
        # out[f"{col}_sd"] = sd
        out[f"{col}_CoV"] = sd / mean if mean != 0 else None
        # out[f"{col}_mad"] = df[col].mad()
        # out[f"{col}_iqr"] = df[col].quantile(0.75) - df[col].quantile(0.25)
    return pd.Series(out)


sample_var = (
    rpdfs
    .groupby(grpby_cols)
    .apply(lambda x: compute_variability(x, features), include_groups=False) 
) # type: ignore
variability_features = sample_var.columns.to_list()
sample_var = sample_var.reset_index()

# # agg samples 
# sample_var.groupby("treatment")[variablility_features].agg(["mean", "std", "median"])

# reduce samples - taking mean of dendrites by animal
animal_var = (
    sample_var[grpby_cols_reduced+variability_features]
    .groupby(grpby_cols_reduced)
    .mean(numeric_only=True)
    .reset_index()
)

# run stats
####################################################################################
ttestDf = utils_stats.merge_mwu_with_ttest_run(
    utils_stats.batch_ttest_run(
        animal_var.melt(id_vars=grpby_cols_reduced), 
        y_var='value', iter_col='variable',
        treatment_col='treatment', group1='young', group2='old'
    ),
    utils_stats.batch_mwu_run(animal_var, variability_features, grp1=None, grp2=None, treatment_col='treatment')
)
print('by_syn stats')
utils_stats.summarize_batch_stats(ttestDf)


# by dend var
####################################################################################
# run within aninimal, treatment dend property variability
dend_var = (
    sumdf
    .groupby(grpby_cols_reduced)
    .apply(lambda x: compute_variability(x, ['count_per_um', 'area', 'intensity_mean']), include_groups=False) 
    
) # type: ignore
variability_features_dend = dend_var.columns.to_list()
dend_var = dend_var.reset_index()

ttestDf_var_dendProps = utils_stats.merge_mwu_with_ttest_run(
    utils_stats.batch_ttest_run(
        dend_var.melt(id_vars=grpby_cols_reduced), 
        y_var='value', iter_col='variable',
        treatment_col='treatment', group1='young', group2='old'
    ),
    utils_stats.batch_mwu_run(dend_var, variability_features_dend, grp1=None, grp2=None, treatment_col='treatment')
)
print('byDend stats')
utils_stats.summarize_batch_stats(ttestDf_var_dendProps)

# plot - boxplots 
####################################################################################
VAR_PLOT_TYPES = ['bySyn', 'byDend']
VAR_PLOT_TYPE_PARAMS = {
    'bySyn':{'data':animal_var, 'metrics_to_plot':variability_features}, 
    'byDend':{'data':dend_var, 'metrics_to_plot':variability_features_dend},
}
SAVE_VARIABILITY_BOXPLOTS = bool(1)
_huevar = 'treatment'


for VAR_PLOT_TYPE in VAR_PLOT_TYPES:
    # Melt for FacetGrid
    _varParams = VAR_PLOT_TYPE_PARAMS[VAR_PLOT_TYPE]
    metrics_to_plot = _varParams['metrics_to_plot']
    plot_df = (
        _varParams['data'][grpby_cols_reduced + metrics_to_plot]
        .melt(id_vars=grpby_cols_reduced, 
            var_name="feature", value_name="value")
        .replace(relabel_keys)
    )
    import glasbey
    pal = dict(zip(plot_df[_huevar].unique(), glasbey.create_palette(palette_size=plot_df[_huevar].nunique())))

    # Create grid
    g = sns.FacetGrid(
        plot_df,
        col="feature",
        col_wrap=3,
        sharey=False,
        height=2,
    )

    # Boxplot + swarm together
    g.map_dataframe(
        sns.boxplot,
        x="treatment",
        y="value",
        hue='treatment', palette=PALETTE_LEG,
        order=PALETTE_LEG.keys(),
        showfliers=False, 
        fill=False,
        linewidth=2,
        linecolor='k',
    )
    g.map_dataframe(
        sns.swarmplot,
        x="treatment",
        y="value",
        hue=_huevar,
        palette={k:'k' for k in PALETTE_LEG},
        order=PALETTE_LEG.keys(),
        size=5, 
    )

    g.set_titles("{col_name}")
    g.set_axis_labels("", "Coefficient of Variation")
    g.figure.tight_layout()
    up.make_legend(PALETTE_LEG, _huevar, kind='patch', ncol=1, bbox_to_anchor=(1, 1))

    if SAVE_VARIABILITY_BOXPLOTS:
        up.save_fig(os.path.join(FIGURE_OUTDIR, f'variability_within_dendrites_{VAR_PLOT_TYPE}.svg'))
        ttestDf.to_csv(os.path.join(FIGURE_OUTDIR, f'variability_within_dendrites_{VAR_PLOT_TYPE}_stats.csv'), index=False)
    plt.show()




import statsmodels.formula.api as smf
import pandas as pd

def lmem_pseudo_r2(fit):
    # fixed-effect part
    X_fe = fit.model.exog        # design matrix for fixed effects
    beta = fit.fe_params.values  # fixed-effect coefficients
    mu_fixed = X_fe @ beta       # fixed-effect prediction
    
    var_fixed = np.var(mu_fixed, ddof=1)

    # random effects:
    # animal variance from cov_re (1x1 if random intercept)
    var_animal = float(fit.cov_re.iloc[0, 0])

    # batch variance from vcomp (first variance component)
    # (check len(fit.vcomp) if you add more vc's)
    var_batch = float(fit.vcomp[0]) if hasattr(fit, "vcomp") else 0.0

    var_random = var_animal + var_batch

    # residual variance
    var_resid = fit.scale

    var_total = var_fixed + var_random + var_resid

    r2_marginal = var_fixed / var_total
    r2_conditional = (var_fixed + var_random) / var_total

    return {
        "var_fixed": var_fixed,
        "var_animal": var_animal,
        "var_batch": var_batch,
        "var_resid": var_resid,
        "R2_marginal": r2_marginal,
        "R2_conditional": r2_conditional,
    }

def fit_all_features_with_r2(df, features):
    rows = []
    for feat in features:
        formula = f"{feat} ~ C(treatment)"
        model = smf.mixedlm(
            formula,
            data=df,
            groups=df["anid"],
            vc_formula={"batch": "0 + C(batch)"},
        )
        fit = model.fit(reml=True)
        stats = lmem_pseudo_r2(fit)
        stats["feature"] = feat
        stats["converged"] = fit.converged
        stats["treatment_effect"] = fit.params.get("C(treatment)[T.young]", np.nan)
        rows.append(stats)
    return pd.DataFrame(rows)

results = []
for feat in variability_features:
    # formula = f"{feat} ~ C(treatment) + C(batch)"  # fixed effects
    formula = f"{feat} ~ C(treatment)"  # batch now in random effects
    
    model = smf.mixedlm(
        formula=formula,
        data=sample_var,
        groups=sample_var["anid"],   # random intercept per animal
        vc_formula={"batch": "0 + C(batch)"},  # random intercept per batch
    )
    fit = model.fit(reml=True)

    # Store a compact summary row
    res = {"feature": feat, 'converged':fit.converged}
    res.update(fit.params.to_dict())
    results.append(res)

results_df = pd.DataFrame(results)
print(results_df)

# Example:
r2_df = fit_all_features_with_r2(sample_var, variability_features)
# print(r2_df)




#############################################################################################
# dendrite type clustering 
#############################################################################################
CLUSTERING = 'here'

# determine variability of regionprop params and determine which capture unique information
#############################################################################################
def determine_feature_uniqueness(X, features):
    """ determine auto-correlation of regionprop params and determine which capture unique information"""
    import seaborn as sns
    import matplotlib.pyplot as plt
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X = X[features].dropna()
    corr = X.corr()

    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Feature Correlation Matrix")
    plt.show()

    sns.clustermap(corr, cmap="coolwarm", vmin=-1, vmax=1, figsize=(10,10))
    # up.save_fig(
    #             os.path.join(r"D:\BygraveLab\Confocal data archive\Pascal\SEGMENTATION_DATASETS\2025_0928_hpc_psd95_tilescans_round4\outputs\2025_1104_clusteringLikeSethGrantPaper", 
    #                          f"feature_uniqueness_clustermap.svg"))
    plt.show()

    vif = pd.DataFrame()
    vif["feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif.sort_values("VIF"))

    return corr , vif


roi_features = [
    # 'roi_intensity_min',
    # 'roi_intensity_max', 
    # 'roi_intensity_mean', 
    # 'roi_axis_major_length',
    # 'roi_area_um',
    # 'roi_intensity_mean_PV',
    # 'axis_area_ratio',
    # 'count',
    # 'count_per_dist',
    # 'count_per_um',
    
    'area',
    # 'intensity_max',
    # 'intensity_mean',
    # 'intensity_min',
    'intensity_std',
    'kurtosis',
    'skewedness',
    'solidity',
]

xCorr, variance_inflation_factor = determine_feature_uniqueness(sumdf, roi_features)


from Analysis.Classification import DataSchema, SchemaResolver, ClassificationPipeline
schema = DataSchema(
    id_cols=["anid", "colocal_id", "roi_i", "batch", "treatment", "ex_i", 'label'],
    group_cols=[],            # <- cluster per genotype (WT vs KO)
    # cat_cols=["sex"],      # <- encode these
    cat_cols=[],      # <- consider excluding
    num_cols=roi_features,
)
config = {
    "encoder": {"name": "one_hot", "params": {"drop": None}},
    "preprocessor": {"name": "standard_scale", "params": {}},
    "clusterer": {"name": "kmeans", "params": {"n_clusters": 6, "random_state": 0}},
    # "clusterer": {"name": "clara_pam", "params": {"n_clusters": 5, "random_state": 0, "subsample_size":50_000}}, # took 2 hours
    "grouping": {"name": "within_groups", "params": {}},
}

pipeline = ClassificationPipeline.from_config(schema, config)
result, meta = pipeline.run(
    (
    # sumdf
    rpdfs
    [schema.id_cols + schema.num_cols])
)
print(result.head())
print(meta)

# display clusters per animal - check clusters are well distributed
print(result.groupby(['anid'])['cluster_label'].value_counts())
result.groupby(['treatment'])['cluster_label'].nunique()
result.groupby(['anid'])['cluster_label'].nunique()

sns.scatterplot(
    data=result,
    x='cluster_score_0',
    y='cluster_score_1',
    hue='treatment',
    style='cluster_label'
)
plt.show()

dfs=[]
for dfn,adf in result.groupby(['anid', 'treatment']):
    ntot = len(adf)
    df = adf['cluster_label'].value_counts().to_frame().reset_index()
    df['tot']=len(adf)
    df['proportion'] = df['count']/df['tot']
    df['anid'] = dfn[0]
    df['treatment'] = dfn[1]
    dfs.append(df)

nperclust = pd.concat(dfs,ignore_index=True)

sns.barplot(data=nperclust,x='cluster_label',y='proportion',hue='treatment', fill=False)
sns.swarmplot(data=nperclust,x='cluster_label',y='proportion',hue='treatment', dodge=True, legend=False)
plt.show()



# synapse types along a individual dendrite
###########################################
# find dends that have different types 
clustersInRois = result.groupby(['treatment', 'ex_i', 'roi_i', 'cluster_label']).size().reset_index()

clustersInRois.query("ex_i==12")
rpdf_cl = pd.merge(rpdfs, result[['ex_i', 'roi_i', 'label', 'cluster_label']], on=['ex_i', 'roi_i', 'label'] )
cz,cy,cx = [],[],[]
for _, row in rpdf_cl.iterrows():
    pass
    z,y,x = ast.literal_eval(row['centroid'])
    cz.append(z)
    cy.append(y)
    cx.append(x)
rpdf_cl = rpdf_cl.assign(**{'cz':cz, 'cy':cy, 'cx':cx})

PALETTE_CLUSTER = dict(sorted(palette_cat(rpdf_cl, 'cluster_label').items()))

# getexi = 1
for getexi in rpdf_cl['ex_i'].unique():
    subdf = rpdf_cl.query("ex_i == @getexi")
    nroi=subdf['roi_i'].nunique()
    roi_is=list(subdf['roi_i'].unique())
    fig,axs=plt.subplots(nroi,2, figsize=np.array((5,3))*np.array((1,nroi)), width_ratios=(0.6,0.4))
    for roi_i, denddf in subdf.groupby('roi_i'):

        dendclustcomp = denddf.groupby(['ex_i', 'roi_i', 'cluster_label']).size().reset_index().rename(columns={0:'count'})
        
        axrow = axs[roi_is.index(roi_i)]
        sns.scatterplot(data=denddf, x='cx', y='cy', hue='cluster_label', ax=axrow[0], palette=PALETTE_CLUSTER, legend=False)
        axrow[0].axis('off')
        axrow[1].pie(
            dendclustcomp['count'], labels=dendclustcomp['cluster_label'], autopct='%1.0f%%', colors=[PALETTE_CLUSTER[l] for l in dendclustcomp['cluster_label']]
        )
    plt.suptitle(f"ex_i={getexi} treatment={subdf['treatment'].values[0]}")
    plt.tight_layout()
    up.save_fig(os.path.join(FIGURE_OUTDIR, f"synapseTypesAlongDends_exi{getexi}.svg"))
    plt.show()


sns.scatterplot(
    data=result,
    x='cluster_label',
    y='roi_intensity_mean_PV',
    hue='treatment',
    style='batch'
)
plt.show()

result.query("cluster_label==1")

# dist plot
############
sns.displot(
    result, 
    x='cluster_score_0',
    y='cluster_score_1',

    row="cluster_label",
    hue="cluster_label",
    col="treatment",
    # hue="treatment", 
    # hue_order=['CON', 'KIR'][::-1],
    height=3, 
    aspect=1.0,
    facet_kws = dict(
        sharex=True,  # keep x-axis consistent
        sharey=True,   # keep y-axis consistent
    ),
    kind='hist',
)

# Create FacetGrid: one row per roi_i
g = sns.FacetGrid(
    result, 
    row="cluster_label",
    # col='roi_i',
    hue="treatment", 
    # hue_order=['CON', 'KIR'],
    height=3, 
    aspect=1.2,
    sharex=True,  # keep x-axis consistent
    sharey=True   # keep y-axis consistent
)

# Map scatterplot to the grid
g.map_dataframe(
    sns.scatterplot,
    x='cluster_score_0',
    y='cluster_score_1',
    # facecolor=None,
    edgecolor='k',
    alpha=0.3,
    
)
        


####################################################################################################################
if bool(0):# PLOT imgs
    ex2roi_map_path = os.path.join(project.project_path, "good_dends.csv")
    ex2roi = pd.read_csv(ex2roi_map_path)
    ex2roimap = {}
    for rowi, row in ex2roi.iterrows():
        print(row)
        ex_i, get_roi_is = row['ex_i'], row['dend_lbls']
        if pd.isnull(get_roi_is): continue
        get_roi_is = ast.literal_eval(get_roi_is)
        ex2roimap[ex_i]=get_roi_is

    all_imgs, all_titles,all_masks = [],[],[]
    for ex in project.examples[:]:
        if int(ex.name) not in ex2roimap.keys():
            continue
        print(ex.name)
        dends_filt = uip.imread(ex.get_path('annotated_dends_filt_ch0.tiff'))
        intensity = uip.imread(ex.get_path('pred_n2v2.tiff'))[0,0]
        dends_filt = uip.filter_label_img(dends_filt, ex2roimap[int(ex.name)])

        # mask intensity
        cropped = {}
        for dend_i in uip.unique_nonzero(dends_filt):
            crops = uip.find_extent_and_crop([dends_filt==dend_i, intensity[0], intensity[1]])
            cropped[dend_i]=crops

        for k,v in cropped.items():
            _pltimgs = v
            _pltimgs = [uip.mip(a) for a in _pltimgs]
            dend_sum_df = sumdf.query(f'(ex_i=={int(ex.name)}) & (roi_i=={k})')
            cpm = np.nan if len(dend_sum_df) == 0 else round(dend_sum_df['count_per_um'].values[0],2)
            rau = np.nan if len(dend_sum_df) == 0 else round(dend_sum_df['roi_area_um'].values[0],2)
            _titles = [f"[{ex.name}] dend {k} count_per_um: {cpm} roi_area_um: {rau}"] + ['']*2
            all_imgs.extend(_pltimgs)
            all_titles.extend(_titles)
            all_masks.extend([None] + [uip.mip(v[0])]*2)
        
    up.plot_image_grid(
        all_imgs, masks=all_masks, titles=all_titles, n_cols=3,
        mask_display_function_kwargs={ 'iterations': 0,'linewidth': 1,'label_cmap': False,'alpha': 0.2,'c_func': lambda x: (1, 0, 0, 0.3) },
        outpath=os.path.join(project.project_path, 'segmentation_summary_for_all_selcectdendsannotated_dends_filt_ch0.svg')
    )
        

