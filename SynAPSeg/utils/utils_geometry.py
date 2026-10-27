from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import (MultiPolygon as shapelyMultiPolygon, Polygon as shapelyPolygon)
from typing import List, Literal
import pandas as pd



def get_poly_intx(polygon_coords, boundary_coords):
    """Return the *intersection* of two polygons (as a Shapely object)."""
    poly = polygon_coords if isinstance(polygon_coords, shapelyPolygon) else shapelyPolygon(polygon_coords)
    boundary = boundary_coords if isinstance(boundary_coords, shapelyPolygon) else shapelyPolygon(boundary_coords)
    return poly.intersection(boundary)


def constrain_polygon_to_boundary(polygon: shapelyPolygon, boundary: shapelyPolygon) -> List[np.ndarray]:
    """Clip *polygon* to *boundary*, returning exterior coordinate arrays."""
    constrained = polygon.intersection(boundary)
    if constrained.is_empty:
        return []
    if isinstance(constrained, shapelyMultiPolygon):
        return [np.asarray(p.exterior.coords) for p in constrained.geoms]
    return [np.asarray(constrained.exterior.coords)]


def get_empty_geojson_feature():
    return {
        "type": "Feature",
        "properties": {"measurements": {}},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[], [], [], [], []]],
        },
    }

def bbox_to_geojson_feature(bbox):
    """ convert a bbox to a geojson feature """
    r1, r2, c1, c2 = bbox

    feature = get_empty_geojson_feature()
    feature['geometry']['coordinates'] = [[[c1, r1], [c2, r1], [c2, r2], [c1, r2], [c1, r1]]]
    return feature


def bbox_hole_to_geojson_feature(shape, bbox):
    """ convert a bbox within a larger bbox (shape) to a geojson feature """
    r1, r2, c1, c2 = bbox
    rows, cols = shape

    feature = get_empty_geojson_feature()
    feature['geometry']['coordinates'] = [
                [[0, 0], [cols, 0], [cols, rows], [0, rows], [0, 0]],  # outer
                [[c1, r1], [c2, r1], [c2, r2], [c1, r2], [c1, r1]],     # hole
            ]
    return feature

    


def view_crop_detections(objects_img, intensity_img, xy_point, ch=0, patch_size=512):
    """Utility for ad‑hoc Napari‑like previewing – *no heavy deps* here."""
    from SynAPSeg.utils import utils_image_processing as uip
    from SynAPSeg.utils import utils_plotting as up

    if len(xy_point) == 2:
        zoom_cc = np.array([xy_point[0], xy_point[0] + patch_size, xy_point[1], xy_point[1] + patch_size]).astype("int32")
    else:
        zoom_cc = xy_point.astype("int32")

    up.show(
        up.overlay_colored_outlines(
            objects_img[ch][zoom_cc[0]: zoom_cc[1], zoom_cc[2]: zoom_cc[3]],
            uip.to_8bit(intensity_img[ch][zoom_cc[0]: zoom_cc[1], zoom_cc[2]: zoom_cc[3]]),
        )
    )


def get_count_df(df: pd.DataFrame, grouping_cols: List[str], area_col: str = "region_area_mm") -> pd.DataFrame:
    """Return per‑group counts + density (counts / area). replaced pool_centroids_by_region"""
    group_counts = df.groupby(grouping_cols).size().rename("count")
    group_means = df.groupby(grouping_cols).mean(numeric_only=True)
    return (
        group_means.join(group_counts).reset_index().assign(count_per_mm=lambda d: d["count"] / d[area_col])
    )

def santize_napari_shapes_features_df(df: pd.DataFrame) -> pd.DataFrame:
    # sanitize feature columns 
    feat_cols = ['name', 'type', 'custom']
    for col in feat_cols:
        if col not in df.columns:
            df[col] = ''  # or some default value
        else:
            df[col] = df[col].fillna('')  # replace NaN with empty string
    
    features_df = df.drop_duplicates(subset='index').set_index(['index'])[feat_cols]
    return features_df

def parse_shapes_from_dataframe(df: pd.DataFrame, coord_system: Literal['matrix', 'cartesian']='matrix') -> tuple[list[np.ndarray], pd.DataFrame]:
    """
    Parse exported napari shapes layer into polygon coordinates and features from a pandas DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing shape data. Must have an 'index' column.
        coord_system (Literal['matrix', 'cartesian']): Coordinate system of the shapes. 
            - 'matrix': Matrix coordinates (row, column). # for napari
            - 'cartesian': Cartesian coordinates (x, y). # for shapely

    Returns:
        tuple[list[np.ndarray], pd.DataFrame]: A tuple containing:
            - poly_coords (list[np.ndarray]): List of polygon coordinates.
            - features_df (pd.DataFrame): DataFrame containing shape features.
    """
    assert 'index' in df.columns, f"DataFrame must have an 'index' column for proper handling."

    reverse_axes = False if coord_system == 'matrix' else True
    axis_cols = sorted([c for c in df.columns if c.startswith('axis-')], reverse=reverse_axes)
    assert len(axis_cols) >= 2, f"DataFrame must have at least two axis columns (found: {axis_cols})."

    # extract polygons and thier coordinates as list of numpy arrays for napari shapes layer    
    df = df.sort_values(['index', 'vertex-index'])
    poly_coords = []
    for dfn, adf in df.groupby(['index']):
        coords = np.stack([adf[c].to_numpy() for c in axis_cols], -1)
        poly_coords.append(coords)

    features_df = santize_napari_shapes_features_df(df)
    
    return poly_coords, features_df

def shapely_to_geojson(geom, properties: dict | None = None) -> dict:
    """
    Wrap a Shapely geometry as a GeoJSON Feature.
        Note: for compatibility with geojsonPoly / ABBA, properties must include measurements key which defines:
            - reg_id = properties['measurements']['ID']
            - region_name = properties['classification']['names'][1]
            - reg_side = properties['classification']['names'][0]
    """
    from shapely.geometry import mapping, shape
    
    return {
        "type": "Feature",
        "properties": properties or {"measurements": {}},
        "geometry": mapping(geom)
    }


def resolve_multipolygon_overlap(geom, mode="priority", eps=1e-4):
    """Resolves overlapping shells in a MultiPolygon.

    mode='merge'    : Dissolves overlapping shells into a single Polygon.
    mode='discard'  : Subtracts the intersection from all shells.
    mode='priority' : Earlier shells clip later shells with an epsilon gap.
    """
    if geom.geom_type == "Polygon":
        return geom

    geoms = [shapely.make_valid(g) for g in geom.geoms]

    if mode == "merge":
        return shapely.unary_union(geoms)  # Unary union of all components

    elif mode == "discard":
        overlap = geoms[0].intersection(geoms[1])
        cleaned = [g.difference(overlap) for g in geoms]
        return shapelyMultiPolygon(cleaned)

    elif mode == "priority":
        cleaned = [geoms[0]]
        cumulative_mask = geoms[0].buffer(eps)
        for g in geoms[1:]:
            trimmed = g.difference(cumulative_mask)
            cleaned.append(trimmed)
            cumulative_mask = cumulative_mask.union(g.buffer(eps))
        return shapelyMultiPolygon(cleaned)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def resolve_overlaps_by_priority(geometries, min_area=1e-4):
    """Resolves overlapping geometries sequentially based on list order.

    - geometries[0] has highest priority and remains unchanged.
    - geometries[i] is clipped by the union of all prior geometries [0 ... i-1].
    """
    resolved = []
    accumulated_union = None

    for i, geom in enumerate(geometries):
        if not geom.is_valid:
            geom = shapely.make_valid(geom)

        if accumulated_union is None:
            cleaned = geom
            accumulated_union = geom
        else:
            # 1. Cut away any area already claimed by higher-priority layers
            cleaned = geom.difference(accumulated_union)

            # 2. Extract polygonal parts if difference generated a GeometryCollection
            #    (which happens when boundaries touch along shared line segments)
            if cleaned.geom_type == "GeometryCollection":
                poly_parts = [
                    g
                    for g in cleaned.geoms
                    if g.geom_type in ("Polygon", "MultiPolygon")
                ]
                cleaned = (
                    shapely.unary_union(poly_parts) if poly_parts else shapelyPolygon()
                )

            # 3. Discard microscopic slivers or completely consumed shapes
            if cleaned.area < min_area:
                cleaned = shapelyPolygon()

            # 4. Accumulate the full footprint for downstream clipping
            accumulated_union = shapely.unary_union([accumulated_union, geom])

        resolved.append(cleaned)

    return resolved

def napari_shapes_to_atlas_geojsonPolys(df, ont, name2ontidx_col: str="acronym", BASE_POLY_NAME: str | None = None, make_nonbase_polys_exclusive=False) -> 'polyCollection':
    """
    Convert Napari shapes layer to atlas geojsonPolys collection.
    
    Args:
        df (pd.DataFrame): DataFrame containing shape data.
        ont (Ontology): Ontology object.
        name2ontidx_col (str): Column name in ontology to use for mapping region names to ontology indices.
    
    Returns:
        polyCollection: Collection of atlas geojsonPolys.
    """
    from shapely.geometry import mapping, shape
    from SynAPSeg.Plugins.ABBA.core_regionPoly import geojsonPoly, polyCollection
    
    
    poly_coords, features_df = parse_shapes_from_dataframe(df, coord_system='cartesian')
    structs_df = ont.source
        
    geojson_features = []
    for name, adf in features_df.groupby('name'):
        
        ont_info = structs_df[structs_df[name2ontidx_col] == name]
        
        if ont_info.empty:
            print(f"Could not find {name} in the ontology")
            raise ValueError(f"Could not find {name} in the ontology")
        
        reg_id = ont_info['id'].values[0]
        region_name = ont_info['name'].values[0]
        acronym = ont_info['acronym'].values[0]
        st_level = ont.ont_ids[reg_id]['st_level']
        
        polygons = []
        for idx in adf.index:
            coords = poly_coords[idx]
            # Close ring if needed
            if not np.array_equal(coords[0], coords[-1]):
                # print(f"Closing ring for {idx}. coords[0], coords[-1]: {coords[0], coords[-1]}")
                coords = np.vstack((coords, coords[0]))
            polygons.append(shapely.make_valid(shapelyPolygon(coords)))


        if len(polygons) > 1:
            geom = resolve_multipolygon_overlap(shapelyMultiPolygon(polygons), mode='priority')
        else:
            geom = polygons[0]  
        
                
        geojsonFeatureElement = {
            "type": "Feature",
            "properties": {
                "napari_shapes_layer_info":{
                    "name": name,
                    "indicies": adf.index.to_list(),
                    "shape_types": adf['type'].to_list(),
                    'custom': adf['custom'].to_list(),
                    'coordinate_system': 'cartesian', # create note that will need to reverse axes in future if loading into napari 
                    
                },
                "measurements": {
                    'ID': reg_id, 
                },
                "classification": {
                    'names': ['left', acronym], # region_side, region_name
                    'st_level': st_level,
                    'acronym': acronym,    # name
                    'region_name': region_name,
                    'name': name
                }
            },
            "geometry": geom
        }
        geojson_features.append(geojsonFeatureElement)
    
    # resolve overlaps between 'exclusive' regions
    if make_nonbase_polys_exclusive:
        if BASE_POLY_NAME is None:
            raise ValueError("BASE_POLY_NAME must be specified when make_nonbase_polys_exclusive is True")
        non_base_gjf = [gjf for gjf in geojson_features if gjf['properties']['classification']['name'] != BASE_POLY_NAME]
        base_gjf = [gjf for gjf in geojson_features if gjf['properties']['classification']['name'] == BASE_POLY_NAME]
        
        resolved_non_base = resolve_overlaps_by_priority([gjf['geometry'] for gjf in non_base_gjf], min_area=1e-4)
        
        
        non_base_gjf_resolved = []
        for gjf, geom in zip(non_base_gjf, resolved_non_base):
            gjf['geometry'] = geom
            non_base_gjf_resolved.append(gjf)
        geojson_features = non_base_gjf_resolved + base_gjf
        
    geojson_polys = []
    for geojson_feature in geojson_features:
        
        geojson_feature['geometry'] = mapping(geojson_feature['geometry'])
        
        gjp = geojsonPoly(
            geojson_feature,
            obj_i=None,  # gets auto-assigned by polyCollection 
            reg_id=geojson_feature['properties']['measurements']['ID'],
            region_name=geojson_feature['properties']['classification']['region_name'],
            reg_side=geojson_feature['properties']['classification']['names'][0],
            st_level=geojson_feature['properties']['classification']['st_level'],
            acronym=geojson_feature['properties']['classification']['acronym'],    # name
            name=geojson_feature['properties']['classification']['name'],
            roi_i=1,
        )
        geojson_polys.append(gjp)
    
    polys = polyCollection(geojsonPolyObjs=geojson_polys)
    
    return polys


from typing import Optional, Tuple, Union
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import shapely
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
    MultiPolygon,
)

def poly_region_prop_longest_path(geom:Union[Polygon, MultiPolygon], um_per_pixel: float = 1.0) -> float:
    """ wrapper for get_longest_polygon_paths function, 
        implemented to allow it to be passed into
        constrain_region_area_by_rois as one of the 
        "region_prop_functions" argument
    """
    length, paths, graph = get_longest_polygon_paths(geom)
    return length * um_per_pixel 
    
def get_longest_polygon_paths(
    geom: Union[Polygon, MultiPolygon],
    density: Optional[float] = None,
    snap_to_boundary: bool = True,
) -> Tuple[float, Optional[Union[LineString, MultiLineString]], nx.Graph]:
  """Computes the longest medial path for every shell in a Polygon or MultiPolygon

  and sums their lengths.

  Parameters
  ----------
  geom : Polygon or MultiPolygon
      Input geometry.
  density : float, optional
      Boundary point spacing. Defaults to (shell perimeter / 250) per part.
  snap_to_boundary : bool
      Whether to snap the terminal path endpoints to the part's exterior shell.

  Returns
  -------
  total_length : float
      The sum of the longest path lengths computed across all shells.
  combined_paths : LineString or MultiLineString
      All per-shell longest paths bundled together.
  full_graph : nx.Graph
      Combined skeleton graph across all components.
  """
  polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]

  full_graph = nx.Graph()
  collected_paths = []
  total_length = 0.0

  for poly in polys:
    if poly.is_empty or poly.area <= 0:
      continue

    # 1. Densify boundary of the current shell
    step = density if density is not None else max(poly.length / 250.0, 1e-4)
    dense_boundary = shapely.segmentize(poly.boundary, max_segment_length=step)
    boundary_pts = MultiPoint(shapely.get_coordinates(dense_boundary))

    # 2. Extract Voronoi ridges inside this shell
    voronoi_lines = shapely.voronoi_polygons(boundary_pts, only_edges=True)
    poly_buffered = poly.buffer(step * 1e-4)
    edges = getattr(voronoi_lines, "geoms", [voronoi_lines])

    part_graph = nx.Graph()
    for edge in edges:
      if isinstance(edge, LineString) and poly_buffered.contains(edge):
        coords = list(edge.coords)
        for i in range(len(coords) - 1):
          u, v = coords[i], coords[i + 1]
          dist = float(np.hypot(v[0] - u[0], v[1] - u[1]))
          part_graph.add_edge(u, v, weight=dist)

    if len(part_graph) == 0:
      continue

    full_graph = nx.compose(full_graph, part_graph)

    # 3. Find diameter path for this shell
    largest_cc = max(nx.connected_components(part_graph), key=len)
    subG = part_graph.subgraph(largest_cc).copy()

    if nx.is_tree(subG):
      root = next(iter(subG.nodes()))
      d1, _ = nx.single_source_dijkstra(subG, root, weight="weight")
      u = max(d1, key=d1.get)
      d2, paths = nx.single_source_dijkstra(subG, u, weight="weight")
      v = max(d2, key=d2.get)
      longest_path = paths[v]
      path_len = d2[v]
    else:
      leaves = [n for n, d in subG.degree() if d == 1] or list(subG.nodes())
      path_len = 0.0
      longest_path = []
      for leaf in leaves:
        dist, paths = nx.single_source_dijkstra(subG, leaf, weight="weight")
        for target in leaves:
          if dist.get(target, 0) > path_len:
            path_len = dist[target]
            longest_path = paths[target]

    if not longest_path:
      continue

    # 4. Snap terminal ends to the shell's exterior boundary
    if snap_to_boundary:
      start_pt = Point(longest_path[0])
      end_pt = Point(longest_path[-1])
      near_start = shapely.ops.nearest_points(poly.exterior, start_pt)[0]
      near_end = shapely.ops.nearest_points(poly.exterior, end_pt)[0]

      path_geom = LineString(
          [near_start.coords[0]] + longest_path + [near_end.coords[0]]
      )
      path_len = path_geom.length
    else:
      path_geom = LineString(longest_path)

    collected_paths.append(path_geom)
    total_length += path_len

  # 5. Bundle paths
  if not collected_paths:
    combined_paths = None
  elif len(collected_paths) == 1:
    combined_paths = collected_paths[0]
  else:
    combined_paths = MultiLineString(collected_paths)

  return total_length, combined_paths, full_graph

def plot_multipolygon_paths(
    geom: Union[Polygon, MultiPolygon],
    G: nx.Graph,
    paths: Optional[Union[LineString, MultiLineString]],
    total_length: float,
):
  """Plots all polygon shells, internal graphs, and the per-shell longest paths."""
  fig, ax = plt.subplots(figsize=(12, 7))
  polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]

  # 1. Draw all shells and holes
  for i, poly in enumerate(polys):
    x, y = poly.exterior.xy
    ax.fill(
        x,
        y,
        alpha=0.12,
        color="#2563eb",
        label="Polygon Shells" if i == 0 else "",
    )
    ax.plot(
        x,
        y,
        color="#1d4ed8",
        lw=1.6,
        label="Exterior Boundaries" if i == 0 else "",
    )

    for interior in poly.interiors:
      ix, iy = interior.xy
      ax.plot(
          ix,
          iy,
          color="#1e40af",
          lw=1.0,
          linestyle="--",
          label="Interior Holes" if i == 0 else "",
      )

  # 2. Draw composite skeleton graph
  if len(G) > 0:
    pos = {node: node for node in G.nodes()}
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="#94a3b8",
        width=0.8,
        alpha=0.6,
        label="Skeleton Graph Edges",
    )

  # 3. Draw every per-shell longest path
  if paths is not None and not paths.is_empty:
    path_list = list(paths.geoms) if isinstance(paths, MultiLineString) else [paths]
    for idx, p in enumerate(path_list):
      px, py = p.xy
      ax.plot(
          px,
          py,
          color="#dc2626",
          lw=3.0,
          zorder=5,
          label="Per-Shell Longest Path" if idx == 0 else "",
      )
      ax.scatter(
          [px[0], px[-1]],
          [py[0], py[-1]],
          c="#fbbf24",
          edgecolors="#111827",
          linewidths=1.2,
          s=110,
          marker="*",
          zorder=6,
          label="Path Terminals" if idx == 0 else "",
      )

  ax.set_aspect("equal")
  ax.grid(True, linestyle=":", alpha=0.5)

  # Deduplicate legend labels
  handles, labels = ax.get_legend_handles_labels()
  by_label = dict(zip(labels, handles))
  ax.legend(
      by_label.values(),
      by_label.keys(),
      loc="upper right",
      framealpha=0.92,
      facecolor="white",
  )

  ax.set_title(
      f"MultiPolygon Skeletons | Total Summed Length: {total_length:.2f}",
      fontsize=13,
      fontweight="bold",
      pad=12,
  )
  plt.tight_layout()
  plt.show()


# --- Quick Test ---
if __name__ == "__main__":
  # Create two distinct polygon shells
  t1 = np.linspace(0, np.pi, 60)
  poly1 = LineString(np.column_stack([t1 * 15, np.sin(t1) * 20])).buffer(3.5)

  t2 = np.linspace(0, 1.8 * np.pi, 80)
  poly2 = LineString(
      np.column_stack([t2 * 10 + 65, np.cos(t2) * 15 - 5])
  ).buffer(3.0)

  multi_poly = MultiPolygon([poly1, poly2])

  total_len, paths, graph = get_longest_polygon_paths(multi_poly)
  print(f"Total Combined Length: {total_len:.4f}")
  plot_multipolygon_paths(multi_poly, graph, paths, total_len)

if __name__ == '__main__': 
    
    # testing napari shapes layer to polygon 
    from SynAPSeg.Plugins.ABBA import utils_atlas_region_helper_functions as arhfs
    df = pd.read_csv(r"D:\OneDrive - Tufts\temp\New folder\2026_0914_AMI_cfosTimecourse\examples\0001\annotated_shapes.csv")
    ont_path = r"D:\OneDrive - Tufts\Classes\Rotation\BygraveLab\SynAPSeg\SynAPSeg\Plugins\ABBA\kim_mouse_10um_v1.1_structures.csv"
    ont = arhfs.Ontology(pd.read_csv(ont_path))
    polys = napari_shapes_to_atlas_geojsonPolys(df, ont, 'acronym')
    
        