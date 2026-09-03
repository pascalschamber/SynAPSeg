import os
import pandas as pd

from SynAPSeg.Plugins.ABBA import core_regionPoly as rp
from SynAPSeg.Plugins.ABBA import utils_atlas_region_helper_functions as arhfs
from SynAPSeg.utils.utils_geometry import (
    bbox_to_geojson_feature, 
)

testdata_base_dir = os.path.abspath('./test_data')
print(testdata_base_dir)

def test_constrain_region_area_by_rois():
    """ test that constraining a multipolygon by taking the intersection using a bbox corresponding to it's boundary returns an object with the correct area (should be identical) """
    ont = arhfs.Ontology(pd.read_csv(os.path.join(testdata_base_dir, 'kim_mouse_10um_v1.1_structures.csv')))
    testregionPolys = rp.polyCollection(os.path.join(testdata_base_dir, 'test_geojson_region_features.geojson'), ont=ont)

    reg_poly = testregionPolys.polygons[1]
    tb = reg_poly.get_bounds(reg_poly.exts[0])
    test_bounds = tuple([tb[2], tb[3], tb[0], tb[1]])
    test_roibbox_polycollection = rp.polyCollection(geojsonPolyObjs=[rp.geojsonPoly(bbox_to_geojson_feature(test_bounds), reg_id=1)])
    test_itx = reg_poly.to_shapely().intersection(test_roibbox_polycollection[0].to_shapely())
    
    passed = round(reg_poly.to_shapely().area, 5) == round(test_itx.area, 5)
    print(f"passed: {passed}\n\treg_poly.to_shapely().area, test_itx.area", reg_poly.to_shapely().area, test_itx.area)
    return passed

def compare_region_counts(df1, df2, grouping_cols=None, value_cols=None):
    """ 
    compare values between two dataframes 

    args:
        df1, df2 : dataframe with regions and counts
        grouping_cols: columns which define the unique regions to compare
            defaults to ['roi_i', 'colocal_id', 'poly_index', 'region_sides', 'reg_id']
        value_cols: columns to calculate difference, nans are converted to 0
            defaults to ['count', 'density']

    """
        
    if grouping_cols is None:
        grouping_cols = ['roi_i', 'colocal_id', 'poly_index', 'region_sides', 'reg_id']
    if value_cols is None:
        value_cols = ['count', 'density']
    
    _df1 = df1[grouping_cols + value_cols].copy()
    _df2 = df2[grouping_cols + value_cols].copy()

    # fill nans in value cols with 0
    _df1[value_cols] = _df1[value_cols].fillna(0)
    _df2[value_cols] = _df2[value_cols].fillna(0)

    # merge on grouping cols
    merged_df = pd.merge(_df1, _df2, on=grouping_cols, suffixes=('_1', '_2'))
    
    # calculate difference
    for c in value_cols:
        merged_df[f'diff_{c}'] = merged_df[f'{c}_1']-merged_df[f'{c}_2']
    
    # sumarrize differences
    summary_strs = ['region counts differences detected (0 indicates no differences):']
    for c in value_cols:
        summary_strs.append(f"{c}: {(merged_df[f'diff_{c}'] != 0).sum()}/{len(merged_df)}")
    summary_str = '\n'.join(summary_strs)
    print(summary_str)
    
    return merged_df



if __name__ == '__main__':
    test_constrain_region_area_by_rois()
