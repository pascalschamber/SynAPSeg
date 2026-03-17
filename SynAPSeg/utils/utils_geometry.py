from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPolygon as shapelyMultiPolygon
from shapely.geometry import Polygon as shapelyPolygon
from typing import List
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



