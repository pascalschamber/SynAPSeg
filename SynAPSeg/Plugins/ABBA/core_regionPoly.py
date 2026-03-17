import sys
from pathlib import Path
import os
import numpy as np
import pandas as pd
import time
import geojson
import timeit
from tifffile import imread
from skimage.measure import regionprops_table
import numba as nb
import shutil
import matplotlib.patches as mpatches
from pathlib import Path
from timeit import default_timer as dt
from numba import jit, types
from numba.typed import List as numbaList
from typing import Dict, Any, Optional, Iterable, List, Tuple
from shapely.geometry import Polygon as shapelyPolygon, MultiPolygon as shapelyMultiPolygon, mapping, shape
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors

from SynAPSeg.utils import utils_general as ug
from SynAPSeg.utils.utils_geometry import get_empty_geojson_feature


# point in polygon functions
##################################################################################################################
def assign_point_in_poly_numba(rpdf, geojsonPolyCollection):
    """    
        fast point in polygon function designed for large data - some edge cases may not be fully handled
            utilizes numba for execution of ray tracing function on deconstructed geojson feature collection polygons 
                such as those exported from qupath/ABBA workflows

    """

    centroids = np.array([c[::-1] for c in rpdf["centroid"].to_list()]) # convert centroids from image to geometric coordinates 

    # iter detections assign to roi_i's
    roi_nb_singles, roi_nb_multis, roi_infos = separate_polytypes(geojsonPolyCollection)
    roi_pp_result = nb_process_polygons(roi_nb_singles, roi_nb_multis, centroids) # return indicies of polygon containing point
    print(np.sum(roi_pp_result))
    
    # extract roi_i from assigned poly - indexing into info and using reg_id as roi_i
    roi_reg_ids = [roi_infos[roi_poly_i]['reg_id'] for roi_poly_i in roi_pp_result]
    assert len(roi_reg_ids) == len(centroids)

    rpdf_final = (
        pd.DataFrame(list(np.array(roi_infos + [{k: np.nan for k in roi_infos[0]}])[roi_pp_result]))
        .assign(centroid_i=np.arange(len(centroids)))
    )
    return rpdf_final


@nb.jit(nopython=True)
def nb_process_polygons(singles, multis, centroids):
    """
    localize centroids to regions, first checking singles (no excluded sub-regions) then multipolygons (have subregions to exclude)
    input numba dtypes:
        ListType[ListType[array(float64, 2d, C)]], ListType[ListType[ListType[array(float64, 2d, C)]]], array(float64, 2d, C)
    returns:
        array(int64, 1d) indicies of polygons containing point. values of -1 indicate point is not in any polygon
    """
    num_polys = len(singles) # singles and multis must have same length
    num_centroids = len(centroids)
    results = np.full(num_centroids, -1)
    for i in nb.prange(num_centroids):
        centroid_x, centroid_y = centroids[i]
        found = False
        for polyi in range(num_polys): 
            if found: 
                break
            these_single_polys, these_multi_polys = singles[polyi], multis[polyi]

            if len(these_single_polys)>0:
                for spi in range(len(these_single_polys)):
                    if nb_point_in_single(these_single_polys[spi], centroid_x, centroid_y):
                        results[i] = polyi
                        found = True
                        break
            if found: 
                break
            if len(these_multi_polys)>0:
                for mpi in range(len(these_multi_polys)):
                    if nb_point_in_multi(these_multi_polys[mpi], centroid_x, centroid_y):
                        results[i] = polyi
                        found = True
                        break
    return results


@nb.jit(nopython=True)
def ray_casting(x, y, centroid_x, centroid_y):
    """determine if point in polygon using ray casting algorithm """
    num_vertices = len(x)
    j = num_vertices - 1
    odd_nodes = False
    for i in range(num_vertices):
        if (
            (y[i] < centroid_y and y[j] >= centroid_y)
            or (y[j] < centroid_y and y[i] >= centroid_y)
            ) and (x[i] <= centroid_x or x[j] <= centroid_x):
            
            # The ^ operator does a binary xor
            # a ^ b will return a value with only the bits set in a or in b but not both
            odd_nodes ^= (
                x[i] + (centroid_y - y[i]) / (y[j] - y[i]) * (x[j] - x[i]) < centroid_x
            )  
        j = i
    return odd_nodes


@nb.jit(nopython=True)
def nb_point_in_single(single, centroid_x, centroid_y):
    main_x, main_y = single.T[0], single.T[1]
    if ray_casting(main_x, main_y, centroid_x, centroid_y):
        return True
    return False

@nb.jit(nopython=True)
def nb_point_in_multi(multi, centroid_x, centroid_y):
    main_polygon, interiors = multi[0], multi[1:]
    main_x, main_y = main_polygon.T[0], main_polygon.T[1]
    if ray_casting(main_x, main_y, centroid_x, centroid_y):
        in_interior = False
        for interior in interiors:
            interior_x, interior_y = interior.T[0], interior.T[1]
            if ray_casting(interior_x, interior_y, centroid_x, centroid_y):
                in_interior = True
                break
        if not in_interior:
            return True
    return False

def get_empty_single_nb():
    # Create a typed list for the outermost level of single polygons (ListType[array(float64, 2d, C)])
    inner_array_type = types.Array(dtype=types.float64, ndim=2, layout='C')
    return numbaList.empty_list(inner_array_type)

def get_empty_multi_nb():
    # Create a typed list for the outermost level of multipolygons (ListType[ListType[array(float64, 2d, C)]])
    inner_array_type = types.Array(dtype=types.float64, ndim=2, layout='C')
    return numbaList.empty_list(types.ListType(inner_array_type))


# main processing functions
##################################################
def constrain_region_area_by_rois(region_polys, roi_regionPolys, um_per_pixel:float=1.0):
    """
    Intersect each region polygon with each ROI polygon and compute constrained
    region areas and corresponding clipped polygons.

    For every (region, ROI) pair, this function:
      • Computes the shapely intersection  
      • Records pixel, µm², and mm² areas along with region metadata  
      • Converts non-empty intersections back into geojson-style polygon objects  

    Parameters
    ----------
    region_polys : list
        Region polygon objects supporting `.to_shapely()`, `.get_total_area()`,
        and standard region metadata attributes.
    roi_regionPolys : list
        ROI polygon objects supporting `.to_shapely()` and `.reg_id` (ROI ID).
    um_per_pixel : float, optional
        Pixel size in microns for area conversion.

    Returns
    -------
    roi_poly_df : pandas.DataFrame
        Per-intersection area metrics and region metadata.
    constrained_regionPolys : dict[int, polyCollection or None]
        Mapping ROI ID → polyCollection of constrained polygons (or None if
        no intersections exist).
    """

    roi_poly_df = []
    _constrained_regionPolys = {roiPoly.reg_id:[] for roiPoly in roi_regionPolys} # dict mapping roi_i to constrained geojsonPoly objects
    for ageojsonPoly in region_polys:
        for aroipoly in roi_regionPolys:
            roipoly = aroipoly.to_shapely()
            regpoly = ageojsonPoly.to_shapely()

            # two different? methods of getting area, seem to be largely similar??
            regpoly_c = regpoly.intersection(roipoly) # TODO make sure polygon holes and multi polys are handled

            roi_i = aroipoly.reg_id

            roi_poly_df.append(dict(
                roi_i=roi_i,
                poly_index=ageojsonPoly.obj_i,
                region_sides=ageojsonPoly.reg_side,
                st_level = ageojsonPoly.st_level,
                region_name = ageojsonPoly.name,     # using extracted region name from ont lookup instead of abba's exported region_name which is just the acronym
                reg_id=ageojsonPoly.reg_id,
                acronym=ageojsonPoly.acronym,
                og_region_area_px=ageojsonPoly.get_total_area(),
                region_area_px = regpoly_c.area,
                region_area_um = pixel_to_um(regpoly_c.area, pixel_size_in_microns=um_per_pixel),
                region_area_mm = pixel_to_mm(regpoly_c.area, pixel_size_in_microns=um_per_pixel),
                # itx_polyObj = regpoly_c,
            ))
            
            # convert itx shapely poly back to geojson poly
            constrained_gjp = None
            if regpoly_c.area > 0:
                constrained_gjp = from_shapely(regpoly_c, obj_i=ageojsonPoly.obj_i, reg_id=ageojsonPoly.reg_id,  region_name=ageojsonPoly.region_name, reg_side=ageojsonPoly.reg_side, st_level=ageojsonPoly.st_level, acronym=ageojsonPoly.acronym, roi_i=roi_i)
            
            _constrained_regionPolys[roi_i].append(constrained_gjp) # TODO need to create geojson poly from this constrained poly

    roi_poly_df = pd.DataFrame(roi_poly_df)

    # # filter out constrained polys with no area and convert remaining to polycollection
    constrained_regionPolys = {}
    for _roi_i, _cgjp_list in _constrained_regionPolys.items():
        filtered_gjps = [el for el in _cgjp_list if el is not None] or None
        
        if filtered_gjps is not None:
            _filtered_gjps = polyCollection()
            _filtered_gjps.add_polys(filtered_gjps)
            filtered_gjps = _filtered_gjps
        
        constrained_regionPolys[_roi_i] = filtered_gjps

    return roi_poly_df, constrained_regionPolys


# handling geojson objects
##################################################################################################################
class polyCollection:
    def __init__(self, geojson_path=None, ont=None, geojsonPolyObjs=None, root_reg_id=997):
        self.geojson_path = geojson_path
        self.polygons = []
        self.root_reg_id = root_reg_id

        if self.geojson_path:
            self.load_objects(self.geojson_path, ont=ont)
        elif geojsonPolyObjs:
            for obj_i, gjp in enumerate(geojsonPolyObjs):
                gjp.obj_i = obj_i # TODO this overrides objects attr, if want to preserve, init empty collection first then use .add_polys()
                gjp.coords_to_array()
                self.polygons.append(gjp)
    
    def load_objects(self, geojson_path, ont=None):
        uobjs = load_geojson_objects(geojson_path)
        self.polygons = []
        for obj_i, obj in enumerate(uobjs):
            gjp = geojsonPoly(obj, debug=False)

            # set attributes 
            attrs = dict(obj_i=obj_i)
            if ont is not None:
                if gjp.reg_id in ont.ont_ids:
                    aE = ont.ont_ids[gjp.reg_id]
                    attrs['st_level'] = aE['st_level']
                    attrs['name'] = aE['name']
                    attrs['acronym'] =  aE['acronym']
                elif gjp.reg_id == self.root_reg_id:
                    attrs['name'] = 'root'
                    attrs['acronym'] =  'root'
                    attrs['st_level'] = 0
                else:
                    raise KeyError(f"gjp.reg_id {gjp.reg_id} not in ont.ont_ids")
                    
            for k,v in attrs.items():
                setattr(gjp, k, v)

            gjp.coords_to_array()
            self.polygons.append(gjp)
    
    def add_polys(self, geojsonPolylist):
        """ add polys to collection without overriding obj_i aka poly index """
        for gjp in geojsonPolylist:
            if gjp.exts is None:
                gjp.coords_to_array()
            self.polygons.append(gjp)
        
    
    def plot(self, polygons=None, ax=None, polypatch_kwargs=None):
        polygons = self.polygons if polygons is None else polygons # for passing a subset e.g. self.plot(polygons = self.polygons[:10])

        no_ax = False
        if ax is None:
            no_ax = True
            fig,ax = plt.subplots()
        
        bounds = []
        for poly in polygons:
            poly: geojsonPoly
            _bounds = poly.plot(ax=ax, polypatch_kwargs=polypatch_kwargs or {})
            bounds.append(_bounds)

        bounds = np.array(bounds)
        MINX, MAXX, MINY, MAXY = bounds[:, 0].min(), bounds[:, 1].max(), bounds[:, 2].min(), bounds[:, 3].max()
        
        if no_ax: 
            # ax.set_xlim(MINX, MAXX)
            # ax.set_ylim(MINY, MAXY)
            ax.relim()
            ax.autoscale_view()
            plt.show()
        else:
            return np.array((MINX, MAXX, MINY, MAXY))
    
    def to_matplotlib_patches(self, geojsonPoly_list, values, cmap=plt.cm.coolwarm, patch_kwargs={}, patchCollection_kwargs={'edgecolors':'k', 'linestyles':'--'}):
        """ convert a list of geojsonPolys to matplotlib patch objects 
            used for creating heatmaps of brain region structures 
        """
        assert len(geojsonPoly_list) == len(values), f"lengths must match but got {len(geojsonPoly_list)}, {len(values)}"
        
        patches = []
        _values = []
        for i, p in enumerate(geojsonPoly_list):
            ptchs = p.to_patch(**patch_kwargs)
            patches.extend(ptchs)
            _values.extend([values[i]]*len(ptchs))

        bound = max(np.abs([min(_values), max(_values)]))
        norm = mcolors.Normalize(
            # vmin=min(densities), vmax=max(densities)
            # vmin=-5, vmax=5,
            vmin=-bound, vmax=bound
        )
        
        patch_collection = PatchCollection(patches, cmap=cmap, norm=norm, **patchCollection_kwargs)
        patch_collection.set_array(np.array(_values))
        return patch_collection
    
    def __iter__(self):
        return iter(self.polygons)

    def __getitem__(self, indices):
        if isinstance(indices, int):
            return self.polygons[indices]
        if isinstance(indices, float):
            return self.polygons[int(indices)]
        
        if not isinstance(indices, tuple):
            indices = tuple(indices)
        return [self.polygons[i] for i in indices]


class geojsonPoly:
    def __init__(self, geojsonFeatureElement, debug=False, **attrs): 
        """
        make it easier to parse the obj dicts return from loading geojson file

        geojsonFeatureElement (dict): e.g. an element from list returned by get_unique_geojson_objs(geojson.load(f)['features'])
        
        attributes
        ``````````
        - properties (dict):
            {
            'objectType': 'annotation',
            'name': 'root',
            'color': [255, 255, 255],
            'classification': {
                'names': ['Left', 'root'], 
                'color': [214, 120, 54]
            },
            'isLocked': True,
            'measurements': {
                'ID': 997.0,
                'Side': 0.0,
                'Atlas_X': 3.386128150979688,
                'Atlas_Y': 3.880648053349701,
                'Atlas_Z': 7.208895310070009
                }
            }

        """
        self.debug = debug
        assert geojsonFeatureElement['type'] == 'Feature'

        self.polytype = geojsonFeatureElement['geometry']['type']
        self.coords = geojsonFeatureElement['geometry']['coordinates']
        self.properties = geojsonFeatureElement.get('properties', {})
        self.flag = ''
        self.n_exteriors = None
        self.n_interiors = None
        self.exts = None # list of (N, 2) arrays
        self.ints = None # list of (M, 2) arrays
        self.ints2ext = None # dict mapping index of exterior poly -> indices of associated interiors
        self.obj_i = None
        self.roi_i = None # used if constraining regions to rois
        
        # ontology related 
        self.reg_id = None
        self.region_name = None
        self.reg_side = None
        try:
            self.reg_id = self.properties['measurements']['ID']
            self.region_name = self.properties['classification']['names'][1]
            self.reg_side = self.properties['classification']['names'][0]
        except:
            pass
        self.st_level = None
        self.acronym = None
        
        # calculated 
        self.region_area = None
        self.region_extent = None
        self.all_obj_atlas_coords = None

        # set attrs 
        for k,v in attrs.items():
            setattr(self, k, v)
            
    def set_coords(self, coords):
        """ sets self.coords, used when constraining a region by an roi """
        self.coords = coords 

    def get_exts(self):
        """ sets exts if coords_to_array not already run TODO: this is a bit clumbsy in that it assumes ints never needed without exts """
        if self.exts is None:
            self.coords_to_array() # need to set .exts #TODO this is a bit clunky here, maybe should be in initialize 
        return self.exts
    
    def to_dict(self):
        """helper function for numba processing, prepares the polygons and extract info needed for centroid df"""

        polyObj_dict = {
            'roi_i': self.roi_i,
            'poly_index':self.obj_i, 
            'reg_id':self.reg_id, 
            'st_level':self.st_level,
            'region_name':self.region_name,
            'reg_side':self.reg_side,
            'acronym':self.acronym,
            'region_area':self.region_area,
            'singles':[], 'multis':[]
        }
        
        for coord_dict in self.to_numba_format():
            if coord_dict['exclude'] is not None:
                polyObj_dict['multis'].append(numbaList([coord_dict['include']] + [a for a in coord_dict['exclude']]))
            else:
                polyObj_dict['singles'].append(coord_dict['include'])
        
        polyObj_dict['singles'] = get_empty_single_nb() if len(polyObj_dict['singles']) == 0 else numbaList(polyObj_dict['singles'])
        polyObj_dict['multis'] = get_empty_multi_nb() if len(polyObj_dict['multis']) == 0 else numbaList(polyObj_dict['multis'])
        return polyObj_dict
    
    def to_numba_format(self):
        """ convert to numba compatible format for fast point in polygon processing """
        l = []
        for i, ext in enumerate(self.get_exts()):
            coord_dict = {'include':ext, 'exclude':None}
            if self.ints and i in self.ints2ext: # check if this polygon exterior has any holes 
                coord_dict['exclude'] = [self.ints[ii] for ii in self.ints2ext[i]]
            l.append(coord_dict)
        return l
    
    def to_region_df_row(self):
        # helper function to convert object to a dict that is compatible with a pandas dataframe
        # note that previously 'region_polygons' were only the first region, now that there are multiple it cannot be used the same way
            # so new implementations that need these coordinate arrays should access them through .numba_format
        atx, aty, atz = self.all_obj_atlas_coords if len(self.all_obj_atlas_coords)==3 else [np.nan]*3
        row_dict = dict(zip(
            ['poly_index', 'region_ids', 'acronym', 'region_name', 'region_sides', 'region_areas', 'region_extents', 'atlas_x', 'atlas_y', 'atlas_z'],
            [self.obj_i, self.reg_id, self.acronym, self.region_name, self.reg_side, self.region_area, self.region_extent, atx, aty, atz]
        ))
        return row_dict
    
        
    
    
    def to_shapely(self):       
        """ convert to shapely polygon """ 
        if self.polytype == 'Polygon':
            return self._make_polygon(0)
        else:
            return shapelyMultiPolygon([self._make_polygon(i) for i in range(len(self.get_exts()))])
    
    def _make_polygon(self, ext_idx) -> shapelyPolygon:
        """ helper fxn for converting to a shapely single polygon """
        exterior = self.get_exts()[ext_idx]
        holes = [self.ints[i] for i in self.ints2ext.get(ext_idx, [])]
        return shapelyPolygon(exterior, holes if holes else None)


    def extract_info(self):
        self.region_area = self.get_total_area()
        self.region_extent = self.get_region_extent()
        self.all_obj_atlas_coords = self.get_atlas_coords() # NEW 2023_0808, not required

    def get_total_area(self):
        """get area of regions to include minus areas to exclude"""
        return sum(polygon_area(a) for a in self.get_exts()) - sum(polygon_area(a) for a in self.ints)
        
    def get_region_extent(self):
        """get bounding box of all polygons comprising this region"""
        return list(get_polygons_extent(self.get_exts() + self.ints))
    
    def get_atlas_coords(self, atlas_measurements_keys=['Atlas_X', 'Atlas_Y', 'Atlas_Z']):
        """ extracts the atlas coords for each region from geojson file if present """
        atlas_coords_not_found = 0 # store number of coords not found
        
        obj_output = []
        measurements = self.properties['measurements']
        for coord_key in atlas_measurements_keys:
            if coord_key not in measurements: # check atlas coords exist
                atlas_coords_not_found += 1
                obj_output.append(None)
            else:
                obj_output.append(measurements[coord_key])
        
        if self.debug and (atlas_coords_not_found > 0):
            print(f'WARN --> atlas coords not extracted (num: {atlas_coords_not_found})', flush=True)
        return obj_output


    def to_geojson_polygon(self):
        """returns a geojson.geometry polygon object - Polygon, MultiPolygon (e.g. geojson.geometry.Polygon)"""
        return getattr(geojson, self.polytype)(self.coords)
    
    def get_coords_gen(self):
        """ Yields the coordinates from a Feature or Geometry """
        return geojson.utils.coords(self.polygon)
    
    def is_2darraylist(self, lst):
        return isinstance(lst[0][0], float)
    


    def parse_polygon_coords(self, polygon_coords):
        """
        convert poly to exterior and interior coords

        Coordinates of a Polygon are an array of linear ring (see
        Section 3.1.6) coordinate arrays.  The first element in the array
        represents the exterior ring.  Any subsequent elements represent
        interior rings (or holes).
        """
        # add scaling feature/function here ! 

        n_interiors = len(polygon_coords) - 1 # a.k.a holes

        ext = np.array(polygon_coords[0])
        ext = self.check_2d_coord_array(ext)

        ints = None
        if n_interiors > 0:
            ints = []
            for ia in polygon_coords[1: ]:
                a = np.array(ia)
                a = self.check_2d_coord_array(a)
                ints.append(a)
        
        return ext, ints


    def coords_to_array(self):
        """
        parses self.coords to separate exteriors and interiors, then runs extract info
            this needs to be run for many functions which depend on access to exts and ints 
        a poly's coords are [ext, int, int ...]
        a multipoly is [poly, poly, ...]
        """
        self.exts = []
        self.ints = []
        self.ints2ext = {}

        if self.polytype == 'Polygon':
            ext, ints = self.parse_polygon_coords(self.coords)
            self.exts.append(ext)
            if ints:
                self.ints.extend(ints)
                self.ints2ext[0] = np.arange(len(ints))
        
        elif self.polytype == 'MultiPolygon':
            for poly_i, poly_coords in enumerate(self.coords):
                ext, ints = self.parse_polygon_coords(poly_coords)
                self.exts.append(ext)

                if ints:
                    self.ints2ext[poly_i] = np.arange(len(ints)) + len(self.ints)
                    self.ints.extend(ints)
                    

        self.n_exteriors = len(self.exts) # this will be >1 if a multipolygon
        self.n_interiors = len(self.ints)  # a.k.a holes
        self.extract_info()
        
    
    def check_2d_coord_array(self, array):
        if array.ndim == 3:
            assert array.shape[0] == 1, f"got shape {array.shape}"
            assert array.shape[2] == 2, f"got shape {array.shape}"
            return array[0]
        elif array.ndim == 2 and array.shape[1] == 2:
            return array
        else: 
            raise ValueError(f"got shape {array.shape}")


    def get_bounds(self, array):
        assert array.ndim == 2
        assert array.shape[1] == 2
        minx, maxx, miny, maxy = array[:, 0].min(), array[:, 0].max(), array[:, 1].min(), array[:, 1].max()
        return (minx, maxx, miny, maxy)
    





    def to_patch(self, **patch_kwargs):
        """
        Convert a to Shapely Polygon or MultiPolygon to a list of matplotlib.patches.Polygon

        Parameters:
        - patch_kwargs: keyword arguments passed to matplotlib.patches.Polygon

        Returns:
        - list of matplotlib.patches.Polygon
        """
            
        # Get exterior coordinates
        shapely_poly = self.to_shapely()
        
        patches = []
        if self.polytype == 'Polygon':
            # Add exterior only
            patches.append(MplPolygon(list(shapely_poly.exterior.coords), closed=True, **patch_kwargs))

        elif self.polytype == 'MultiPolygon':
            for poly in shapely_poly.geoms:
                patches.append(MplPolygon(list(poly.exterior.coords), closed=True, **patch_kwargs))

        else:
            raise TypeError("Input must be a Shapely Polygon or MultiPolygon", str(type(geojsonPoly)))

        return patches

    



    def plot(self, exts=None, ints=None, ax=None, polypatch_kwargs=None):
        import matplotlib.patches as patches
        exts = exts or self.get_exts()
        ints = ints or self.ints

        no_ax = False
        if ax is None:
            no_ax = True
            fig,ax = plt.subplots()

        bounds = []
        for i in range(len(exts)):
            minx, maxx, miny, maxy = exts[i][:, 0].min(), exts[i][:, 0].max(), exts[i][:, 1].min(), exts[i][:, 1].max()
            bounds.append(np.array((minx, maxx, miny, maxy)))
            _polypatch_kwargs = dict(linewidth=0.5, edgecolor='cyan', alpha=1, facecolor='none')
            _polypatch_kwargs.update(polypatch_kwargs or {})
            ax.add_patch(
                patches.Polygon(exts[i],**_polypatch_kwargs)
            )
        
        if ints:
            for i in range(len(ints)):
                minx, maxx, miny, maxy = ints[i][:, 0].min(), ints[i][:, 0].max(), ints[i][:, 1].min(), ints[i][:, 1].max()
                bounds.append(np.array((minx, maxx, miny, maxy)))
                _polypatch_kwargs = dict(linewidth=0.5, edgecolor='red', alpha=1, facecolor='none')
                _polypatch_kwargs.update(polypatch_kwargs or {})
                ax.add_patch(patches.Polygon(ints[i], **_polypatch_kwargs))

        bounds = np.array(bounds)
        pad = 20
        MINX, MAXX, MINY, MAXY = bounds[:, 0].min()-pad, bounds[:, 1].max()+pad, bounds[:, 2].min()-pad, bounds[:, 3].max()+pad
        
        
        if no_ax: 
            ax.set_xlim(MINX, MAXX)
            ax.set_ylim(MINY, MAXY)
            plt.show()
        
        else:
            return np.array((MINX, MAXX, MINY, MAXY))

    
    def __str__(self):
        return (
            f"""reg_id: {self.reg_id} - region_name:{self.region_name} - reg_side:{self.reg_side} - polytype:{self.polytype}
            n_exteriors: {self.n_exteriors}, n_interiors: {self.n_interiors}""" + \
            (f"\n\tflag: {self.flag}" if len(self.flag) > 0 else '')
        )
    def __repr__(self):
        return str(self)

# transforms
#################################################################################
# isotropic scaling
def scale_geojsonPoly(gjp:geojsonPoly, scale_factor=1.0) -> geojsonPoly:
    """ 
        isotropic scaling of region polys 
        working on a *copy* of a geojsonPoly, scale self.coords then rerun self.coords_to_array and return scaled version 
        #TODO only checked if plot looks good, likely more work to be done to make sure area, functionality, etc. is updated/preserved
        # hence why not integrated as class method. Likely first todo would be update self.coords as well 
    """
    from copy import deepcopy
    self = deepcopy(gjp)
    self.exts = []
    self.ints = []
    self.ints2ext = {}

    if self.polytype == 'Polygon':
        ext, ints = self.parse_polygon_coords(self.coords)
        self.exts.append(ext * scale_factor)
        if ints:
            self.ints.extend([i * scale_factor for i in ints])
            self.ints2ext[0] = np.arange(len(ints))

    elif self.polytype == 'MultiPolygon':
        for poly_i, poly_coords in enumerate(self.coords):
            ext, ints = self.parse_polygon_coords(poly_coords)
            self.exts.append(ext * scale_factor)

            if ints:
                self.ints2ext[poly_i] = np.arange(len(ints)) + len(self.ints)
                self.ints.extend([i * scale_factor for i in ints])
                

    self.n_exteriors = len(self.exts) # this will be >1 if a multipolygon
    self.n_interiors = len(self.ints)  # a.k.a holes
    self.extract_info()
    return self

def scale_polyCollection(gjpc: polyCollection, scale_factor=1.0) -> polyCollection:
    """apply scale_geojsonPoly over a list of polys returning a new object"""
    scaled_polys = [scale_geojsonPoly(p, scale_factor) for p in gjpc.polygons]
    scaled_polyColl = polyCollection()
    scaled_polyColl.add_polys(scaled_polys)
    return scaled_polyColl


# converters
#################################################################################
def roi_feature_to_polygon(roi_feature: Dict) -> shapelyPolygon:
    """
    Convert a single GeoJSON Feature (Polygon) to a Shapely Polygon.
    """
    geom = roi_feature.get("geometry", {})
    p = shape(geom)
    # Light repair if necessary
    if not p.is_valid:
        p = p.buffer(0)
    if p.is_empty:
        raise ValueError("ROI polygon is empty after parsing.")
    return p

def shapely_to_geojson(geom, properties: Dict | None = None) -> Dict:
    """
    Wrap a Shapely geometry as a GeoJSON Feature.
        Note: for compatibility with geojsonPoly, properties must include measurements key
    """
    return {
        "type": "Feature",
        "properties": properties or {"measurements": {}},
        "geometry": mapping(geom)
    }

def from_shapely(_shapelyPolygon, obj_i:Optional[int]=None, reg_id:Optional[int]=None, region_name:Optional[str]=None, reg_side:Optional[str]=None, st_level:Optional[int]=None, acronym:Optional[str]=None, roi_i:Optional[int]=None, **attrs) -> geojsonPoly:
    """ convert a shapely polygon object to a geojson poly object """
    geojsonFeatureElement = shapely_to_geojson(_shapelyPolygon)
    
    # geojsonFeatureElement = get_empty_geojson_feature() # old method
    # geojsonFeatureElement['geometry']['type'] = shapelyPolygon.__geo_interface__['type']
    # geojsonFeatureElement['geometry']['coordinates'] = shapelyPolygon.__geo_interface__['coordinates']

    return geojsonPoly(
        geojsonFeatureElement, obj_i=obj_i, reg_id=reg_id,  region_name=region_name, reg_side=reg_side, st_level=st_level, acronym=acronym, roi_i=roi_i, **attrs
    )

# handling parsing napari shapes
#################################################################################
def parse_Napari_shapes_to_polygons(
    df: pd.DataFrame,
    group_col: str = "index",
    vertex_col: str = "vertex-index",
    x_col: str = "axis-5",   # axis-5 if X in STCZYX, if YX fmt would be axis-1
    y_col: str = "axis-4",   # Y in STCZYX
    drop_degenerate: bool = True,
) -> Tuple[Dict, List[shapelyPolygon]]:
    """
    Convert Napari shapes (from dataframe grouped by `group_col`) into a GeoJSON FeatureCollection
    and a list of Shapely Polygons.

    The CSV must contain one row per vertex with columns:
    - group_col: polygon id (e.g., "index")
    - vertex_col: vertex order within polygon (0..N-1)
    - x_col: x coordinate (axis-5, if exported on image of fmt STCZYX)
    - y_col: y coordinate (axis-4)

    Args:
        csv_path: Path to the CSV.
        group_col: Column that identifies polygon groups.
        vertex_col: Column ordering vertices.
        x_col: Column with X coord (axis-5).
        y_col: Column with Y coord (axis-4).
        drop_degenerate: If True, skip invalid/empty/too-small polygons.

    Returns:
        (feature_collection_dict, list_of_shapely_polygons)
    """
    
    # Basic checks
    for col in (group_col, vertex_col, x_col, y_col):
        if col not in df.columns:
            raise ValueError(f"CSV is missing required column: {col}")

    features = []
    polys: List[shapelyPolygon] = []

    # Build one polygon per group, ordered by vertex-index
    for poly_idx, group in df.groupby(group_col):
        g = group.sort_values(vertex_col, kind="mergesort")
        xs = g[x_col].astype(float).tolist()
        ys = g[y_col].astype(float).tolist()

        coords: List[Tuple[float, float]] = list(zip(xs, ys))

        # Ensure at least 3 distinct points
        if len(coords) < 3:
            if drop_degenerate:
                continue
            else:
                coords = coords * 2  # force something (not recommended)

        # Close ring if needed
        if coords[0] != coords[-1]:
            coords.append(coords[0])

        poly = shapelyPolygon(coords)

        # Fix minor self-intersections if any
        if not poly.is_valid:
            poly = poly.buffer(0)

        # Optionally drop degenerate/invalid polygons
        if drop_degenerate and (poly.is_empty or not poly.is_valid or poly.area == 0):
            continue

        polys.append(poly)

        features.append({
            "type": "Feature",
            "properties": {"index": int(poly_idx)},
            "geometry": mapping(poly)
        })

    fc = {"type": "FeatureCollection", "features": features}
    return fc, polys

# handling polygon operations
#################################################################################
def subtract_exclusions_from_roi(roi_poly: shapelyPolygon, exclusion_polys: Iterable[shapelyPolygon]):
    """
    Subtract a list/iterable of shapely polygons from the ROI polygon.

    Returns:
        Shapely geometry (Polygon or MultiPolygon).
    """
    # filter out empty polygons
    exclusion_polys = [p for p in exclusion_polys if p and not p.is_empty]
    if len(exclusion_polys) == 0:
        return roi_poly
    
    # create union of all exclusion polygons
    exclusion_union = unary_union(exclusion_polys)
    if exclusion_union.is_empty:
        return roi_poly
    
    # return difference
    return roi_poly.difference(exclusion_union)


# handling geojson feature collections
#################################################################################
def write_geojson_featureCollection(outpath, featureCollection) -> None:
    """ write a geojson FeatureCollection to disk """
    from geojson import FeatureCollection, dump
    assert isinstance(featureCollection, FeatureCollection)
    with open(outpath, 'w') as f:
        dump(featureCollection, f)


def load_geojson_objects(geojson_path, filter_ABBA_region_features=True, feature_filter_condition=None):
    """
    read .geojson file and parse features to get only valid annotations

    args: 
        geojson_path: path to geojson feature collection
        filter_ABBA_region_features: bool
            applies a filter to get all detections that have an 'ID' (regions), except the root which doesn't have an atlas id
        feature_filter_condition: callable
            filter is applied to each feature 
            note: if filter_ABBA_region_features==True, this is ignored
    
    returns:
        list of geojson feature dicts 
    """
    
    if filter_ABBA_region_features:
        feature_filter_condition = lambda obj: ('measurements' in obj['properties'].keys()) and ('ID' in obj['properties']['measurements'])

    with open(geojson_path) as f:
        allobjects = geojson.load(f)
    allfeatures = allobjects['features']
    if len(allfeatures)==0: 
        raise ValueError(f"no features loaded")
    
    if feature_filter_condition is not None:
        allfeatures = [obj for obj in allfeatures if feature_filter_condition(obj)]
        if len(allfeatures) == 0: 
            raise ValueError(f"no region objects found after filtering")
    
    if filter_ABBA_region_features:
        # sanitize objs, removes duplicates if was extracted multiple times without clearing previous regions
        allfeatures = get_unique_geojson_objs(allfeatures)

    return allfeatures

def get_unique_geojson_objs(obj_list):
    """ filter geojson objs by getting unique combos of ID and side, take last instance assuming last is most recent 
            handles case where mulitple exports have been done in qupath
    """
    ud = {}
    for i, obj in enumerate(obj_list):
        id_side = '-'.join(obj['properties']['classification']['names'])
        if id_side not in ud:
            ud[id_side] = []
        ud[id_side].append(i)
    unique_objs = [obj_list[v[-1]] for k,v in ud.items()]
    return unique_objs


def extract_polyregions(geojson_objs, ont):
    # build region polygon objects, parsing single and multipolygons from qupaths geojson file
    all_polys = []
    for obj_i, anobj in enumerate(geojson_objs):
        print_str = ""

        aPoly = regionPoly(obj_i=obj_i, anobj=anobj, st_level=get_st_lvl_from_regid(anobj['properties']['measurements']['ID'], ont))
        aPoly.region_name = ont.ont_ids[aPoly.obj_id]['name'] if aPoly.obj_id != 997 else 'root'

        coord_arrs, num_poly_coords = get_coordinates(anobj)
        print_str += f"{obj_i} ({anobj['geometry']['type']}) --> n: {num_poly_coords}\n"
        try:
            maybeError = aPoly.unpack_feature_polygons(coord_arrs)
            aPoly.extract_info()
            
            if maybeError is not None: 
                raise(ValueError)
            
        except ValueError:
            print(aPoly.print_str)
            raise ValueError(f'ERROR:\n{print_str}\n{maybeError}')
        
        finally:
            all_polys.append(aPoly)
    return all_polys

def get_st_lvl_from_regid(regid, ont):
    st_lvl = ont.ont_ids[regid]['st_level'] if regid in ont.ont_ids else 0
    return st_lvl

def separate_polytypes(polyObjs):
    """
    build initial numba format for singles and multipolygons
        
        numba input needs to be consistent, such that:
            each element of input list is all polys for a given region
            this element will contain two lists, one for singles and one for multis
                can be multiple multipolygons so nested again, is empty if none
                    for each element of the inner list the first element is coords to include and rest are excluded

    Returns:
        - singles: ListType[ListType[array(float64, 2d, C)]], 
        - multies: ListType[ListType[ListType[array(float64, 2d, C)]]], 
        - poly info: list[dict[Any]] returned by polyObj.to_dict()
    """
    # get regionPoly objects as dicts and sort by st_lvl
    polyObj_dicts = sorted([apoly.to_dict() for apoly in polyObjs], key=lambda x: x['st_level'], reverse=True) 
    
    # gather numba capatible outer list and info for each poly
    nb_singles, nb_multis, infos = numbaList(), numbaList(), []
    for di, d in enumerate(polyObj_dicts):
        infos.append({k:v for k,v in d.items() if k not in ['singles', 'multis']}) # get info_only (not arrays)
        nb_singles.append(d['singles'])
        nb_multis.append(d['multis'])

    assert len(nb_singles) == len(nb_multis) == len(infos), f"lengths do not match {len(nb_singles)}, {len(nb_multis)}, {len(infos)}"
    return nb_singles, nb_multis, infos

def get_coordinates(aFeature):
    polyType = aFeature['geometry']['type']
    
    if polyType == 'MultiPolygon':
        coord_arrs = [np.array(el) for el in aFeature['geometry']['coordinates']]
    elif polyType == 'Polygon':
        coord_arrs = [np.array(aFeature['geometry']['coordinates'])]
    else: 
        raise ValueError(f"{polyType} is not handled")
    num_poly_coords = len(coord_arrs)
    
    return coord_arrs, num_poly_coords


class regionPoly:
    """
        TODO: DEPRECATED  ????

        object to hold a collection of polys so know the following:
            regions to extract or ignore
            area
            numba compatible, so can return list of polys (in order so it can be proc'd in order)
        polygon coords are stored in a dict where keys are interally used poly indices (in order)
            value is a dict: 
                arr: np.ndarray, 
                polytype: main_poly, exteriors, interiors
        there are three types of polys in qupath abba annotatins
            a region ('exteriors')
            a multipolygon with regions to exclude ('main')
            those regions to exclude from this main polygon ('interiors')
        Interface with nuclei localization
            this is the order based on what we can infer from the different types of polygons
                doing it this way avoids potential issue when checking multipolys first where it could be in excluded region
                and also in a later non-multipoly
            check against non-multipolys, if inside can be sure not going to be excluded
            then, check multipolys, if inside and not in excluded areas we're good
    ARGS
        GO_FAST (bool): if true skip superfoulous calculations (debugging)
    """

    def __init__(self, obj_i, anobj, st_level, GO_FAST=True):
        self.valid_polytypes = ['main', 'exteriors', 'interiors']
        self.count_polytypes = None # used for debugging
        self.obj_i = obj_i # this is index of polygon collection in geojson file (i.e. poly_index)
        self.st_level = st_level
        self.region_area = None
        self.region_extent = None # store bounding box coordinates
        self.poly_count = 0
        self.poly_arrays = {} # store polys here
        self.GO_FAST = GO_FAST
        self.ingest_obj(anobj)


    def ingest_obj(self, anobj):
        self.anobj = anobj # might want to not store to conserve memory if possible
        self.region_name = None
        self.obj_id = anobj['properties']['measurements']['ID']  # this is id of region in ontology
        # self.obj_names = ', '.join(anobj['properties']['classification']['names'])
        obj_names = anobj['properties']['classification']['names']
        self.reg_side = str(obj_names[0])
        self.acronym = str(obj_names[1])
        self.geometry_type = anobj['geometry']['type']

    def __str__(self):
        prt_str = ''
        get_attrs = ['obj_id', 'acronym', 'reg_side', '', 'region_name',  '', 'geometry_type','', 'region_area']
        for attr in get_attrs:
            if len(attr) == 0: prt_str+='\n'
            else: prt_str += f"{attr}: {getattr(self, attr)} "
        return prt_str+'\n'
    
    def add_poly(self, poly_arr, polyType):
        if polyType not in self.valid_polytypes: 
            raise ValueError(f'polyType ({polyType}) must be one of {self.valid_polytypes}')
        assert self.poly_count not in self.poly_arrays, f"key ({self.poly_count}) should not already exist"

        # add poly to dict
        self.poly_arrays[self.poly_count] = {'arr':poly_arr, 'polyType':polyType}
        self.poly_count += 1
        
    def unpack_feature_polygons(self, coord_arrs):
        # store exteriors and interiors, where exteriors are regions to include, and interiors are regions to exclude
        # store shapes for debugging
        self.ragged_shapes = []
        error = None
        self.print_str = ""
        self.numba_format = [] # list of dicts, where dict includes indicies to include/ exclude for each coord array
        
        try:
            for arr_i, arr in enumerate(coord_arrs):
                nb_dict = {'include':None, 'exclude':None}
                if arr.ndim == 3: # array of shape e.g. (1, 2, nPoints)
                    if arr.shape[0]>1: raise ValueError(f"{arr.shape} is not handled")
                    for v_idx in range(arr.shape[0]): # but could handle it if implemented here
                        valid_arr = arr[v_idx]
                        assert valid_arr.ndim ==2, f"{valid_arr.shape} is not 2d"
                        if arr_i == 0: # handle case where only a single poly
                            self.add_poly(valid_arr, 'main')
                        else:
                            self.add_poly(valid_arr, 'exteriors')
                        nb_dict['include'] = valid_arr
                        
                elif arr.ndim == 1: # handle ragged arrays 
                    unpacked_arrs = [np.array(el) for el in arr]
                    self.ragged_shapes.append(f"{arr.shape} --> {[a.shape for a in unpacked_arrs]}")
                    
                    # check_all_2dim
                    unpacked_shapes = [el.shape for el in unpacked_arrs]
                    assert all([len(el)==2 for el in unpacked_shapes]), f"{unpacked_shapes} contains non 2d arrays"
                    nb_dict['exclude'] = []
                    # split into main body and interiors
                    for i, el in enumerate(unpacked_arrs):
                        if i == 0:
                            self.add_poly(el, 'main')
                            nb_dict['include'] = el
                        else:
                            self.add_poly(el, 'interiors')
                            nb_dict['exclude'].append(el)
                else: 
                    raise ValueError(f'this should not happen, arr ndim: {arr.ndim}')
                self.numba_format.append(nb_dict)
            

            self.print_str += f"{self.obj_i} ({self.geometry_type}) --> n: {len(coord_arrs)}\n"
            for astr in self.ragged_shapes:
                self.print_str += f"\tragged arr: {astr}\n"
            
        except Exception as e:
            error = e

        finally: # append additional info
            for arr in coord_arrs:
                self.print_str += f'{arr.shape}'
                for ca_i, ca in enumerate(coord_arrs):
                    if ca.ndim == 1:
                        for el in ca:
                            self.print_str += f'\t {np.array(el).shape}'
        
        return error
    
    def prepare_numba_input(self, polygonCollection):
        # where polygonCollection is a list of dict with keys for include and exclude 
        # and include is an array and  exclude is list of coord arrays
        nb_include, nb_exclude = [], []
        for d in polygonCollection:
            exclude = nb.typed.numbaList(d['exclude']) if d['exclude'] is not None else None
            nb_include.append(d['include']), nb_exclude.append(exclude)
        return nb.typed.numbaList(nb_include), nb.typed.numbaList(nb_exclude)

    def extract_info(self):
        self.region_area = self.get_total_area(self.poly_arrays)
        self.region_extent = self.get_region_extent(self.poly_arrays)
        self.all_obj_atlas_coords = self.get_atlas_coords(self.anobj) # NEW 2023_0808, not required
        if not self.GO_FAST:
            self.get_count_polytypes()
    

    def get_count_polytypes(self):
        # count num polys of each type, for plotting/debuging
        self.count_polytypes = dict(zip(self.valid_polytypes, [0]*len(self.valid_polytypes)))
        for pi, p_arr in self.poly_arrays.items():
            polyType = p_arr['polyType']
            self.count_polytypes[polyType]+=1
            
    def get_total_area(self, poly_arrays):
        # get area of regions to include minus areas to exclude
        total_area = 0
        for poly_i, poly_dict in poly_arrays.items():
            area = polygon_area(poly_dict['arr']) 
            area *= -1 if poly_dict['polyType'] == 'interiors' else 1
            total_area += area
        return total_area
    
    def get_region_extent(self, poly_arrays):
        """get bounding box of all polygons comprising this region - returns minimum_x, minimum_y, maximum_x, maximum_y """
        return list(get_polygons_extent([pd['arr'] for pd in poly_arrays.values()]))
    
    def get_atlas_coords(self, geojson_obj):
        # extracts the atlas coords for each region from geojson file if present
        atlas_coords_not_found = 0 # store number of coords not found
        atlas_measurements_keys = ['Atlas_X', 'Atlas_Y', 'Atlas_Z']
        obj_output = []
        measurements = geojson_obj['properties']['measurements']
        for coord_key in atlas_measurements_keys:
            if coord_key not in measurements: # check atlas coords exist
                atlas_coords_not_found += 1
                obj_output.append(None)
            else:
                obj_output.append(measurements[coord_key])
        if not self.GO_FAST:
            if atlas_coords_not_found > 0: print(f'WARN --> atlas coords not extracted (num: {atlas_coords_not_found})', flush=True)
        return obj_output
    
    def to_dict(self):
        # helper function for numba processing, prepares the polygons and extract info needed for centroid df
        polyObj_dict = {
            'poly_index':self.obj_i, 
            'reg_id':self.obj_id, 
            'st_level':self.st_level,
            'region_name':self.region_name,
            'reg_side':self.reg_side,
            'acronym':self.acronym,
            'region_area':self.region_area,
            'singles':[], 'multis':[]}
        for coord_dict in self.numba_format:
            if coord_dict['exclude'] is not None:
                polyObj_dict['multis'].append(numbaList([coord_dict['include']] + [a for a in coord_dict['exclude']]))
            else:
                polyObj_dict['singles'].append(coord_dict['include'])
        
        polyObj_dict['singles'] = get_empty_single_nb() if len(polyObj_dict['singles']) == 0 else numbaList(polyObj_dict['singles'])
        polyObj_dict['multis'] = get_empty_multi_nb() if len(polyObj_dict['multis']) == 0 else numbaList(polyObj_dict['multis'])
        return polyObj_dict
    
    def to_region_df_row(self):
        # helper function to convert object to a dict that is compatible with a pandas dataframe
        # note that previously 'region_polygons' were only the first region, now that there are multiple it cannot be used the same way
            # so new implementations that need these coordinate arrays should access them through .numba_format
        atx, aty, atz = self.all_obj_atlas_coords if len(self.all_obj_atlas_coords)==3 else [np.nan]*3
        row_dict = dict(zip(
            ['poly_index', 'region_ids', 'acronym', 'region_name', 'region_sides', 'region_areas', 'region_extents', 'atlas_x', 'atlas_y', 'atlas_z'],
            [self.obj_i, self.obj_id, self.acronym, self.region_name, self.reg_side, self.region_area, self.region_extent, atx, aty, atz]
        ))
        return row_dict

# general utils
##############################################
def print_obj(geojson_obj):
    """ print str representation of a geojson obj without coordinates"""
    prt_str = ''
    for k,v in geojson_obj.items():
        if not isinstance(v, dict):
            astr = f"\t{v}"
        else:
            if k == 'geometry':
                print_d = {kk:vv for kk,vv in v.items() if kk != 'coordinates'}
            else:
                print_d = {kk:vv for kk,vv in v.items()}
            astr = ''
            for kk, vv in print_d.items():
                astr += f"\t{kk}:\n\t\t{vv}\n"

        prt_str += f"{k}:\n{astr}\n"
    print(prt_str)


def polygon_area(region_poly):
    ''' Calculates the area of a complex polygon using the shoelace formula. Calculates signed area, so abs is taken to get the actual area '''
    assert region_poly.shape[1] == 2 and region_poly.ndim==2, f'error --> region poly shape {region_poly.shape}'
    Xs, Ys = region_poly[:,1], region_poly[:,0]
    area = 0.5 * abs(sum(Xs[i]*(Ys[i+1]-Ys[i-1]) for i in range(1, len(Xs)-1)) + Xs[0]*(Ys[1]-Ys[-1]) + Xs[-1]*(Ys[0]-Ys[-2]))
    return area

def pixel_to_mm(pixel_area, pixel_size_in_microns):
    # Convert pixels to square millimeters
    pixel_area_in_mm = (pixel_size_in_microns / 1000) ** 2 * pixel_area
    return pixel_area_in_mm

def pixel_to_um(pixel_area, pixel_size_in_microns):
    # Convert pixels to square um
    pixel_area_in_um = (pixel_size_in_microns**2) * pixel_area
    return pixel_area_in_um

def get_polygons_extent(poly_list):
    """ returns minimum_x, minimum_y, maximum_x, maximum_y of all polys in a given region """
    minimum_x, minimum_y, maximum_x, maximum_y = np.inf, np.inf, 0, 0
    
    for arr in poly_list:
        if len(arr) == 0:
            continue
        max_x = np.max(arr[:,0])
        max_y = np.max(arr[:,1])
        min_x = np.min(arr[:,0])
        min_y = np.min(arr[:,1])
        if max_x > maximum_x:
            maximum_x = max_x
        if max_y > maximum_y:
            maximum_y = max_y
        if min_x < minimum_x:
            minimum_x = min_x
        if min_y < minimum_y:
            minimum_y = min_y
    return minimum_x, minimum_y, maximum_x, maximum_y


# plotting helpers
##############################################
def plot_polygons(apoly, plot_points=None, fig_title=None, show_legend=True, invert=True, 
                  region_outlines=None, region_label=False, 
                  SAVE_PATH=None, limit_plot=True, fig_size=(20,20), 
                  fc_alpha=0.2, pad=200, input_ax=None
):
    if input_ax is None:
        fig,ax = plt.subplots(1, figsize=fig_size)       
    else: 
        ax = input_ax
        
    
    if apoly is None:
        # e.g. for ploting all region outlines instead of region patches
        assert isinstance(region_outlines, list)
        minimum_x, minimum_y, maximum_x, maximum_y = 0, 0, np.inf, np.inf
        x_min, x_max = minimum_x-pad, maximum_x+pad
        y_min, y_max = minimum_y-pad, maximum_y+pad
    else: # support for passing a single array of coordinates too
        if isinstance(apoly, regionPoly):
            minimum_x, minimum_y, maximum_x, maximum_y = get_polygons_extent([pd['arr'] for pd in apoly.poly_arrays.values()])
        elif isinstance(apoly, np.ndarray):
            minimum_x, minimum_y, maximum_x, maximum_y = get_polygons_extent([apoly])
            poly_dict = {0:{'arr':apoly, 'polyType':'main'}}
        else: 
            raise ValueError()

        # limit plot area
        x_min, x_max = minimum_x-pad, maximum_x+pad
        y_min, y_max = minimum_y-pad, maximum_y+pad
        if limit_plot:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)


    # plot outlines of all regions provided, expects a list of polyRegions
    if region_outlines is not None: # put these down first so current polylist is clear       
        fc_alpha= 0.2 if fc_alpha is None else fc_alpha 
        for ap in region_outlines:
            for pi,pd in ap.poly_arrays.items():
                centroid = pd['arr'].mean(axis=0)
                if not(x_min <= centroid[0] <= x_max and y_min <= centroid[1] <= y_max):
                    continue
                ax.add_patch(patches.Polygon(pd['arr'], linewidth=0.5, edgecolor='cyan', alpha=0.15, facecolor='none'))
                
                if region_label is False: 
                    continue
                # add region label
                c_lbl = ap.anobj['properties']['classification']['names'][1]
                bbox_props = dict(boxstyle="square,pad=0.3", fc="black", ec="black", lw=1, alpha=0.7)
                ax.text(centroid[0], centroid[1], c_lbl, ha='right', va='top', bbox=bbox_props, c='w', fontsize='xx-small')
    
    if apoly is not None:
        # Add the patch to the Axes
        fc_alpha= 0.2 if fc_alpha is None else fc_alpha
        blue_rgba = (0.0, 0.0, 1.0, fc_alpha)    # with opacity
        green_rgba = (0.0, 1.0, 0.0, fc_alpha)  
        red_rgba = (1.0, 0.0, 0.0, fc_alpha)    
        yellow_rgba = (1.0, 1.0, 0.0, 0.5)
        palette2 = {'exteriors':yellow_rgba, 'main': yellow_rgba, 'interiors':(1.0, 0.0, 0.0, 0.5)}
        palette2_fc = {'exteriors':blue_rgba, 'main': green_rgba, 'interiors':red_rgba}
        
        to_add = apoly.poly_arrays if isinstance(apoly, regionPoly) else poly_dict
        for pi, pd in to_add.items():
            arr, ptype = pd['arr'], pd['polyType']
            ec, fc = palette2[ptype], palette2_fc[ptype]
            polygon = patches.Polygon(arr, linewidth=0.5, edgecolor=ec, facecolor=fc)
            ax.add_patch(polygon)

    if plot_points is not None:
        ax.scatter(plot_points[:,0], plot_points[:,1], c='k', s=25, marker='x')
    

    if fig_title is not None: plt.title(fig_title)
    
    if show_legend and region_outlines is None:
        # Create a legend using the provided color palettes
        legend_handles= [mpatches.Patch(edgecolor=palette2[label], facecolor=palette2_fc[label], label=label, alpha=0.2) for label in palette2]
        ax.legend(handles=legend_handles, loc='best')
    
    if invert:
        plt.gca().invert_yaxis()

    if SAVE_PATH is None and input_ax is None:
        plt.show()
    elif SAVE_PATH is not None:
        outname = ug.clean_filename(fig_title, replace_spaces='_') + '.svg'
        fig.savefig(os.path.join(SAVE_PATH, outname), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        pass


# debugging
########################################################################################################
def plot_compare_region_counts(region_count_df, test_region_count_df, stain_col_names, st_lvl_max=10):
    import seaborn as sns
    # plot comparison of shared nuclei per region for two dataframes (for debugging)
    shared_reg_ids = list(set(region_count_df['reg_id'].unique()).intersection(set(test_region_count_df['reg_id'].unique())))
    shared_columns = list(set(region_count_df.columns.to_list()).intersection(set(test_region_count_df.columns.to_list())))
    df_gt = region_count_df[region_count_df['reg_id'].isin(shared_reg_ids)][shared_columns].assign(condition='gt')
    df_gt['st_level'] = df_gt['st_level'].replace('notFound', 0)
    df_test = test_region_count_df[test_region_count_df['reg_id'].isin(shared_reg_ids)][shared_columns].assign(condition='test')
    df_plot = pd.concat([df_gt, df_test], ignore_index=True)
    df_plot['st_level'] = df_plot['st_level'].replace('notFound', 0).astype('int')
    
    for cell_type in stain_col_names:
        fig,ax = fig, ax = plt.subplots(figsize=(20,10))
        bp = sns.swarmplot(    
            data=df_plot[df_plot['st_level']>st_lvl_max], x='region_name', y=cell_type, hue='condition', palette=dict(zip(['gt', 'test'], ['k','r'])), ax=ax,
            dodge=True,
        )
        ax.set_xticks(np.arange(len(ax.get_xticklabels())), ax.get_xticklabels(), rotation=-45, ha='left')
        plt.show()

def get_info_index(poly_list, info_LoD, poly_index):
    # lookup polygon by poly_index 
    inds, polys, infos = [], [], []
    for di, d in enumerate(info_LoD):
        if d['poly_index'] == poly_index:
            inds.append(di)
            polys.append(poly_list[di])
            infos.append(d)
    return inds, polys, infos

def plot_coordinates_inside_polygon(rpdf, nb_singles, infos, regionPoly_list, centroids, get_polyi=0, get_colocal_id=0 ):
    # plot nuclei inside a polygon (for debugging)
    # NOTE: only shows points that were assigned the lowest structural level since rpdf only contains these 
    print(len(rpdf[rpdf['poly_index']==get_polyi]))
    get_inds, get_polyarrs, get_info  = get_info_index(nb_singles, infos, get_polyi)
    apoly = [el for el in regionPoly_list if el.obj_i == get_polyi][0]
    get_centroids_df = rpdf[(rpdf['poly_index']==get_polyi) & (rpdf['colocal_id']==get_colocal_id)]
    centroid_coords = centroids[get_centroids_df['centroid_i'].values]
    plot_polygons(apoly, plot_points=centroid_coords, limit_plot=True) 

def plot_unassigned_coordinates(rpdf, centroids, regionPoly_list, get_polyi=0):
    # plot coordinates of nuclei that were not assigned to any region, can show a region too
    # where process_polygons_df is the output of process_polygons function after info has been added back
    get_unassigned_centroids_df = rpdf[(pd.isnull(rpdf['poly_index']))]
    unassigned_centroids = centroids[get_unassigned_centroids_df['centroid_i'].values]
    plot_polygons(regionPoly_list[get_polyi], plot_points=unassigned_centroids, limit_plot=False) 

def print_poly_list(poly_list):
    # print info for a list of regionPolys
    for pi, apoly in enumerate(poly_list):
        print(apoly.region_name)
        print(pi, f'n: {len(apoly.poly_arrays)}')
        
        if bool(1):
            for k,v in apoly.anobj['properties'].items():
                print(f'\t{k}: {v}')
        total_area = 0
        for poly_i, poly_dict in apoly.poly_arrays.items():
            ptype = poly_dict['polyType']
            coords = poly_dict['arr']
            area = polygon_area(coords)
            area *= -1 if ptype == 'interiors' else 1
            total_area += area
            print(poly_i, ptype, coords.shape, f'area: {area}')
        print(f"total area: {pixel_to_um(total_area)}\n")

def inspect_highest_num_polys(all_polys, nmin=0, nmax=5, PLOT=True, SAVE_PATH=None):
    # for each datum get the top most poly regions
    for geojson_path, apolylist in all_polys.items():
        print(geojson_path, len(apolylist))
        numpolys = {}    
        for pi, apoly in enumerate(apolylist):
            npolys = len(apoly.poly_arrays)
            numpolys[pi] = npolys

        most_ps = sorted(numpolys.items(), key=lambda item: item[1])[::-1][nmin:nmax]
        most_ps_idxs = [el[0] for el in most_ps]
        for pidx in most_ps_idxs:
            print(f'datum obj idx: {pidx}')
            print_poly_list(apolylist[pidx:pidx+1])
            if PLOT:
                apoly = apolylist[pidx]
                fig_title = f"{Path(geojson_path).stem}\n[{pidx}] {apoly.obj_names+ ', ' + str(apoly.obj_id)} - num main/exts/ints: {apoly.count_polytypes['main']}/{apoly.count_polytypes['exteriors']}/{apoly.count_polytypes['interiors']} - {apoly.geometry_type} - area: {int(apoly.total_area)}"
                plot_polygons(apoly, fig_title=fig_title, region_outlines=apolylist, SAVE_PATH=SAVE_PATH)
