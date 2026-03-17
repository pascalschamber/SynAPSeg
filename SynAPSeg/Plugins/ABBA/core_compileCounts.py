"""
functions here aid with propogating region counts up atlas heirarchy 
and compiling results from several observations from same subject 
i.e. merging counts across several sections from same animal

EXAMPLE USAGE
--
setup below is used for compiling counts from rpdfs representing mulitple observations per animal

REGION_INDEXERS = ['uExID', 'roi_i', 'region_sides']           # structural indexers (rpdf cols) which define unique sets of region polys (which have different mapping of poly_i to regions)
POPULATION_INDEXERS = ['roi_i', 'colocal_id', 'cluster_label']             # indexers which define populations which may have unique region counts within a structural context

EXP_LUT = map_poly_hierarchy(
    region_df,
    ont,
    REGION_INDEXERS
)

PRP_LUT = Compile.initialize_propagation_container(rpdf_final, EXP_LUT, POPULATION_INDEXERS)
PRP_LUT = Compile.populate_hierarchy_indicies(rpdf_final, EXP_LUT, PRP_LUT, POPULATION_INDEXERS, REGION_INDEXERS)

final_counts = extract_counts(
    rpdf, region_df, PRP_LUT, POPULATION_INDEXERS, REGION_INDEXERS,
    extract_mean_columns, get_region_df_cols
)
    
"""

import pandas as pd
import numpy as np
import SynAPSeg.Plugins.ABBA.utils_atlas_region_helper_functions as arhfs
from SynAPSeg.utils import utils_general as ug
from SynAPSeg.Analysis.df_utils import build_query, query_df, ugroups

def to_unique_list(alist):
    """ helper function to remove non-unique elements from a list, preserving order of elements """
    out = []
    for el in alist:
        if el not in out:
            out.append(el)
    return out

def get_ex_largest_side(countdfs, count_col='count'):
    """ return side with highest sum over `count_col` """
    usides = countdfs['region_sides'].unique()
    if len(usides) ==1:
        return countdfs  # only one side present
    
    ex_side_counts = countdfs.groupby(['ex_i', 'region_sides'])[count_col].sum().reset_index().pivot(index='ex_i', columns='region_sides', values=count_col)
    ex_side_counts['largest'] = ex_side_counts.apply(
        lambda row: 'Right' if row['Right'] > row['Left'] else 'Left', axis=1
    )
    ex_sides = ex_side_counts['largest'].reset_index().rename(columns={'largest':'region_sides'})

    # subset original df
    subset = countdfs.merge(ex_sides, on=['ex_i', 'region_sides'])
    return subset


def aggregate_by_animal(df:pd.DataFrame, grouping_cols, sum_cols, avg_cols, drop_cols=None, count_col='count'):
    """
    Aggregates data by animal (or other groups), performing sums and weighted averages on specified columns.
    
    Args:
        df (pd.DataFrame): The source dataframe.
        grouping_cols (list): Columns to group by (e.g., ['animal_id', 'region']).
        sum_cols (list): Columns to sum directly (e.g., ['count', 'region_area_mm']).
        avg_cols (list): Columns to be weighted by 'count' before averaging.
        drop_cols (list, optional): Columns to exclude from the final output.
        
    Returns:
        pd.DataFrame: The aggregated dataframe with recalculated densities.
    """
    temp_df = df.copy()
    
    # Create weighted values (e.g. Intensity * Count)
    for c in avg_cols:
        temp_df[c] = temp_df[c] * temp_df[count_col]

    # Sum both the bases and the weighted values
    agg_dict = {col: 'sum' for col in (sum_cols + avg_cols)}

    aggregated_df:pd.DataFrame = temp_df.groupby(grouping_cols).agg(agg_dict).reset_index()

    #Divide weighted sums by total count to get true means
    for c in avg_cols:
        aggregated_df[c] = aggregated_df[c] / aggregated_df[count_col]
    
    if drop_cols:
        aggregated_df = aggregated_df.drop(columns=drop_cols, errors='ignore')

    return aggregated_df


def map_poly_hierarchy(region_df, ont, REGION_INDEXERS) -> dict[tuple, dict[int, list[int]]]:
    """
    Creates a `lineage map`: a LUT mapping each unique poly to its parents (and self) for each unique 
        structural context defined by region indexers.
        This functions to define how counts in finer regions need to propogate up the atlas region heirarchy
            ex. count in CA1 contributes to the count in CA, HIP, and HPF regions 
    
        Structure:
            {region_group: {poly_i: [self+parents]}}
        
        TODO: this is slow, better would be to replace repeated queries with vectorization
    """
    # region_df must have these cols
    assert all([c in region_df.columns for c in ['reg_id', 'poly_index']])

    EXP_LUT = {}
    u_poly_inds = ugroups(region_df, REGION_INDEXERS)
    for _, row in u_poly_inds.iterrows():

        bq = {c:row[c] for c in REGION_INDEXERS}
        this_idx = tuple(bq.values())
        if this_idx in EXP_LUT: 
            raise ValueError(this_idx)
        EXP_LUT[this_idx] = {}

        # this is df slice for this index - which has unique poly_is - all base ids
        q_res = region_df.query(build_query(**bq))  

        # build hierarchy
        for _, row in q_res.iterrows():
            base_reg = row['reg_id']
            base_poly_i = row['poly_index']
            parent_reg_ids = arhfs.get_all_parents(ont, base_reg, parent_ids=None)
            reg_poly_is = [base_poly_i] # include self 

            for pri in parent_reg_ids:
                pri_q_res = q_res.query(build_query(reg_id=pri))

                if len(pri_q_res) == 1:
                    reg_poly_is.append(pri_q_res['poly_index'].values[0])
                elif len(pri_q_res) == 0:
                    pass
                elif len(pri_q_res) > 1:
                    raise ValueError(f"len(pri_q_res): {len(pri_q_res)}, query: {build_query(**bq, **{'reg_id':pri})}\npri_q_res:\n{pri_q_res}")

            # # santity check - each id should be child of the next
            # regs = [q_res[q_res['poly_index']==pi]['reg_id'].values[0] for pi in reg_poly_is]
            # for r in regs:
            #     print(r, ONI(r).parents())

            EXP_LUT[this_idx][base_poly_i] = reg_poly_is

    return EXP_LUT

# init and pop funcs below are DEPRECATED in favor of make_PRP_LUT_merged
def initialize_propagation_container(rpdf, lineage_map, population_indexers):
    """
    Sets up a nested dictionary to hold propagated indices for each population and structural context
        Structure: {population_group: structural_group: {poly_i: []}}
        # DEPRECATED
    """
    container = {}
    
    for _, row in ugroups(rpdf, population_indexers).iterrows():
        
        # Create a unique key for the population (e.g., cluster label + colocal ID)
        pop_key = tuple(row[col] for col in population_indexers)
        container[pop_key] = {}
        
        for structural_index, poly_heirarchy in lineage_map.items():
            container[pop_key][structural_index] = {pi:[] for pi in poly_heirarchy.keys()}  

    return container


def populate_hierarchy_indicies(
    rpdf, 
    EXP_LUT, 
    PRP_LUT, 
    POPULATION_INDEXERS,
    REGION_INDEXERS,
    ):
    """
    Iterates through the detection dataframe and propagates row indices 
        from the finest regions up to the parent regions.
        # DEPRECATED
    """
    # Group by the specified indexers to find detections within specific regions
    rpdf_indexer = list(set(POPULATION_INDEXERS).union(REGION_INDEXERS)) + ['poly_index']
    for dfn, adf in rpdf.groupby(rpdf_indexer):
        
        dfnq = dict(zip(rpdf_indexer, dfn)) # this collective structural/population query
        adf_row_inds = list(adf.index)
        structural_index = tuple([dfnq[ri] for ri in REGION_INDEXERS])
        pop_index = tuple([dfnq[ri] for ri in POPULATION_INDEXERS])

        # Get the full lineage (child + parents) for this specific polygon
        parent_poly_is = EXP_LUT[structural_index][dfnq['poly_index']]
        for ppi in parent_poly_is:
            PRP_LUT[pop_index][structural_index][ppi].extend(adf_row_inds)

    return PRP_LUT

def make_PRP_LUT_merged(
    rpdf, 
    EXP_LUT,
    POPULATION_INDEXERS,
    REGION_INDEXERS,
):
    """
        Iterates through the detection dataframe and propagates row indices 
            from the finest regions up to the parent regions.
            this impl merges init and propogation steps to minimize looping
    """
    PRP_LUT = {}
        
    # Group by the specified indexers to find detections within specific regions
    rpdf_indexer = to_unique_list(POPULATION_INDEXERS + REGION_INDEXERS + ['poly_index'])

    for dfn, adf in rpdf.groupby(rpdf_indexer):
        
        dfnq = dict(zip(rpdf_indexer, dfn)) # this collective structural/population query
        
        pop_index = tuple([dfnq[ri] for ri in POPULATION_INDEXERS])
        structural_index = tuple([dfnq[ri] for ri in REGION_INDEXERS])
        adf_row_inds = list(adf.index)

        if pop_index not in PRP_LUT.keys():
            PRP_LUT[pop_index] = {}
        if structural_index not in PRP_LUT[pop_index].keys():
            # init with dict mapping each poly_index to an empty list
            poly_heirarchy = EXP_LUT[structural_index]
            PRP_LUT[pop_index][structural_index] = {pi:[] for pi in poly_heirarchy.keys()}

        # Get the full lineage (child + parents) for this specific polygon
        parent_poly_is = EXP_LUT[structural_index][dfnq['poly_index']]
        for ppi in parent_poly_is:
            PRP_LUT[pop_index][structural_index][ppi].extend(adf_row_inds)
            
    return PRP_LUT

def extract_counts(
        rpdf, region_df, PRP_LUT, 
        POPULATION_INDEXERS,REGION_INDEXERS,
        extract_mean_columns, get_region_df_cols
    ):
    """ propogate counts and extract region props over the heirarchy """
    
    # TODO update LUTs to use single flat index composed of population, structural index and poly i
    count_rows = []
    for pop_index, struct_heirarchy in PRP_LUT.items():
        for struct_index, heirarchy in struct_heirarchy.items():
            for poly_i, rpdf_inds in heirarchy.items():
                pq = dict(zip(POPULATION_INDEXERS, pop_index))
                sq = dict(zip(REGION_INDEXERS, struct_index))
                region_df_row = region_df.query(build_query(**sq, **{'poly_index':poly_i}))

                region_properties = {}
                region_properties['count'] = len(rpdf_inds)
                region_properties['density'] = np.nan           # calculate this after, just holding a spot

                rps = rpdf.loc[rpdf_inds, extract_mean_columns].mean(numeric_only=True).to_dict()

                count_rows.append(ug.merge_dicts(
                    sq, 
                    pq, 
                    region_properties,
                    dict(zip(get_region_df_cols, region_df_row[get_region_df_cols].values[0])),
                    rps
                ))
                
    return pd.DataFrame(count_rows)



# ---------------------------------------------------------------------------
# to test - and create core file which implements these functions
# ---------------------------------------------------------------------------

def fast_populate_hierarchy(rpdf, EXP_LUT, POPULATION_INDEXERS, REGION_INDEXERS):
    """
    Optimized index propagation using vectorized mapping and grouping.

    # TODO: INCOMPLETE - DONT USE
        doesn't handle empty regions properly so propogation will be incomplete
        issue likely in step 3 merge operation 
        so while likely faster implementation, will come back to this 
        
    """
    # 1. Flatten the Expansion LUT into a mapping DataFrame
    # This creates a lookup of: (structural_key, child_poly) -> [parent_polys]
    expansion_data = []
    for structural_index, mapping in EXP_LUT.items():
        for child_poly, parents in mapping.items():
            for p in parents:
                expansion_data.append((*structural_index, child_poly, p))

    # Create a helper DF for merging. Columns will be: [*REGION_INDEXERS, 'poly_index', 'parent_poly_index']
    mapping_cols = REGION_INDEXERS + ['poly_index', 'parent_poly_index']
    mapping_cols = to_unique_list(mapping_cols)
    mapping_df = pd.DataFrame(expansion_data, columns=mapping_cols)

    # 2. Prepare the main dataframe
    # We only need the indexers and the original row index
    _cols = to_unique_list(POPULATION_INDEXERS + REGION_INDEXERS + ['poly_index'])
    work_df = rpdf[_cols].copy()
    work_df['original_index'] = rpdf.index

    # 3. Merge to "Propagate" indices
    # This replaces the 'for ppi in parent_poly_is' loop
    # It joins every row to ALL its parent polygons simultaneously
    merged_df = work_df.merge(
        mapping_df, 
        on= to_unique_list(REGION_INDEXERS + ['poly_index']), 
        how='left'
    )

    # 4. Group and aggregate indices into lists
    # Group by (Population, Structural, Parent_Poly)
    final_groupers = POPULATION_INDEXERS + REGION_INDEXERS + ['parent_poly_index']
    final_groupers = to_unique_list(final_groupers)
    
    # This produces a Series where each entry is a list of original row indices
    aggregated = merged_df.groupby(final_groupers)['original_index'].apply(list)

    # 5. (Optional) Convert back to your specific nested dict format
    # Only do this if downstream code strictly requires the dict structure.
    return aggregated.to_dict()



