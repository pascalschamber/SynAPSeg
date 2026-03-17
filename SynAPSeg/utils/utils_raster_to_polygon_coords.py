import rasterio
import shapely
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Union, Optional


def sort_coordinates_by_distance(
    coordinates: Union[List[Tuple[float, float]], np.ndarray],
    centroid: Optional[Tuple[float, float]] = None
) -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
    """
    Sorts a list of (x, y) coordinates in order of increasing distance from a reference point.
    
    Parameters
    ----------
    coordinates : List[Tuple[float, float]] or np.ndarray
        A list or NumPy array of shape (N, 2), where each element is an (x, y) tuple representing a coordinate.
    centroid : Optional[Tuple[float, float]], optional
        A distinct (x, y) coordinate to be used as the reference point for distance calculations.
        If not provided, the centroid of the input coordinates will be calculated and used.
    
    Returns
    -------
    sorted_coordinates : List[Tuple[float, float]]
        The input coordinates sorted by their distance from the reference point, from closest to farthest.
    used_centroid : Tuple[float, float]
        The (x, y) coordinates of the centroid used for distance calculations. This is either the provided
        centroid or the calculated centroid if none was provided.
    
    Raises
    ------
    ValueError
        If the input list of coordinates is empty.
    TypeError
        If the input is neither a list of tuples nor a NumPy array.
    
    Usage example
    -------------
    # Sample list of coordinates representing an object
    object_coordinates = [(1, 2), (3, 4), (5, 6), (7, 8), (2, 1), (4, 3), (6, 5), (8, 7), (0, 0), (9, 9)]
    
    # Call the function without providing a centroid
    sorted_coords, centroid = sort_coordinates_by_distance(object_coordinates)
    print(f"Calculated Centroid of the object: {centroid}\n")
    print("Coordinates sorted by increasing distance from centroid:")
    for coord in sorted_coords:
        print(coord)
    
    sorted_coords, centroid = sort_coordinates_by_distance(object_coordinates, centroid=(9,9))
    print(f"Calculated Centroid of the object: {centroid}\n")
    print("Coordinates sorted by increasing distance from centroid:")
    for coord in sorted_coords:
        print(coord)
        
    """
    # Input validation
    if isinstance(coordinates, list):
        coords = np.array(coordinates)
    elif isinstance(coordinates, np.ndarray):
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("object coordinates array must be of shape (N, 2).")
        coords = coordinates
    else:
        raise TypeError("coordinates must be a list of tuples or a NumPy array.")
    
    if coords.size == 0:
        raise ValueError("The list of coordinates is empty.")
    
    # Determine the reference centroid
    if centroid is None:
        centroid_x = np.mean(coords[:, 0])
        centroid_y = np.mean(coords[:, 1])
        used_centroid = (centroid_x, centroid_y)
    else:
        if (not isinstance(centroid, tuple)) or (len(centroid) != 2):
            raise TypeError("centroid must be a tuple of two floats representing (x, y).")
        used_centroid = centroid
        centroid_x, centroid_y = used_centroid
    
    # Compute Euclidean distances from each coordinate to the reference centroid
    distances = np.sqrt((coords[:, 0] - centroid_x) ** 2 + (coords[:, 1] - centroid_y) ** 2)
    
    # Get the sorted indices based on distances
    sorted_indices = np.argsort(distances)
    
    # Sort the coordinates based on the sorted indices
    sorted_coords = coords[sorted_indices]
    
    # Convert sorted coordinates back to list of tuples
    sorted_coordinates = [tuple(coord) for coord in sorted_coords]
    
    return sorted_coordinates, used_centroid



    
def assign_labels(
    object_centroids: Union[np.ndarray, List[Tuple[float, float]]],
    polygons_per_label: Dict[int, List[shapely.geometry.Polygon]]
) -> Tuple[List[int], List[int]]:
    """
    Given a list/array of object centroids (x, y) and a dictionary of label->[Polygons],
    return two lists of the same length:
        - assigned_labels: label ID if the point is inside that label's polygon(s), or 0 if no match.
        - polygon_label_subindex_found: index of the polygon within its label's list where the point was found, or -1 if no match.

    Parameters
    ----------
    object_centroids : np.ndarray or List[Tuple[float, float]]
        An array/list of shape (N, 2), each row is (x, y).
    polygons_per_label : Dict[int, List[shapely.geometry.Polygon]]
        A dictionary where the key is the label and the value is a list of polygons
        belonging to that label.

    Returns
    -------
    assigned_labels : List[int]
        A list of label IDs (or 0 if not found) for each centroid.
    polygon_label_subindex_found : List[int]
        A list of polygon sub-indices within their label's polygon list where the point was found, or -1 if not found.
        
    
    Example usage
    -------------
    # Sample object centroids
    object_centroids_ex = np.array([
        (1.5, 1.5),
        (3.0, 3.0),
        (5.0, 5.0),
        (99.0, 3.0)
    ])

    # Sample polygons per label
    polygons_per_label_ex = {
        1: [Polygon([(0, 0), (0, 2), (2, 2), (2, 0)]), Polygon([(2, 2), (2, 4), (4, 4), (4, 2)])],
        2: [Polygon([(2, 2), (2, 4), (4, 4), (4, 2)])],
        3: [Polygon([(4, 4), (4, 6), (6, 6), (6, 4)])]
    }

    # Assign labels to centroids
    labels_ex, polygon_label_subindex_found = assign_labels(object_centroids_ex, polygons_per_label_ex)

    print(labels_ex)  # Output: [1, 2, 3]
    """
    # Ensure object_centroids is a NumPy array for efficient processing
    if isinstance(object_centroids, list):
        object_centroids = np.array(object_centroids)
    elif not isinstance(object_centroids, np.ndarray):
        raise TypeError("object_centroids must be a NumPy array or a list of tuples.")

    if len(object_centroids) == 0:
        return [], []
    
    # Prepare a list of all polygons along with their corresponding labels and sub-indices
    all_polygons = []
    polygon_label_map = []
    polygon_label_subindex = []  # Indices of the polygons within their label key

    for label, polygons in polygons_per_label.items():
        all_polygons.extend(polygons)
        polygon_label_map.extend([label] * len(polygons))
        # Enumerate polygons to track their sub-indices within the label
        polygon_label_subindex.extend(list(range(len(polygons))))

    # Build a spatial index for all polygons
    spatial_index = shapely.strtree.STRtree(all_polygons)

    assigned_labels = []
    polygon_label_subindex_found = []  # List to store the sub-index of the matching polygon

    for centroid in object_centroids:
        point = shapely.geometry.Point(centroid)
        # Query the spatial index for possible containing polygons
        possible_polygons = spatial_index.query(point)  # Assumes this returns polygon indices

        label_found = 0  # Default label if no polygon contains the point
        subindex_found = -1  # Default sub-index if no polygon contains the point

        for polygon_i in possible_polygons:
            polygon = all_polygons[polygon_i]

            if polygon.contains(point):
                # Retrieve the label corresponding to this polygon
                label_found = polygon_label_map[polygon_i]
                # Retrieve the sub-index of this polygon within its label
                subindex_found = polygon_label_subindex[polygon_i]
                break  # Stop after the first matching polygon is found

        assigned_labels.append(label_found)
        polygon_label_subindex_found.append(subindex_found)

    return assigned_labels, polygon_label_subindex_found






def assign_labels_to_object_indices(
    object_indices_list: List[np.ndarray],
    polygons_per_label: Dict[int, List[shapely.geometry.Polygon]]
) -> Tuple[List[int], List[int]]:
    """
    Given a list/array of object indices (x, y) and a dictionary of label->[Polygons],
    return two lists of the same length:
        - assigned_labels: label ID if the object is inside that label's polygon(s), or 0 if no match.
        - polygon_label_subindex_found: index of the polygon within its label's list where the object was found, or -1 if no match.

    Parameters
    ----------
    object_indices : list of np.ndarray corresponding to an array of spatial coordinates for each object
            arrays must be of shape (N, 2), each row is (x, y).
    polygons_per_label : Dict[int, List[shapely.geometry.Polygon]]
        A dictionary where the key is the label and the value is a list of polygons
        belonging to that label.

    Returns
    -------
    assigned_labels : List[int]
        A list of label IDs (or 0 if not found) for each object index.
    polygon_label_subindex_found : List[int]
        A list of polygon sub-indices within their label's polygon list where the object was found, or -1 if not found.
        
    Example usage
    -------------
    # Sample object indices (x, y) in a 2D array
    object_indices_ex = [
        np.array([
        (1.5, 1.5),   # Should be in label 1, sub-index 0
        (3.0, 3.0),   # Could be in label 1, sub-index 1 or label 2, sub-index 0
        (5.0, 5.0),   # Should be in label 3, sub-index 0
        (99.0, 3.0)   # Should have no match
        ]),
        np.array([
        (3.0, 3.0),   
        (5.0, 5.0),   
        (99.0, 3.0)   
        ]),
        np.array([
        (5.0, 5.0),   
        (99.0, 3.0)   
        ])
        ]

    # Sample polygons per label
    polygons_per_label_ex = {
        1: [
            Polygon([(0, 0), (0, 2), (2, 2), (2, 0)]),  # Sub-index 0
            Polygon([(2, 2), (2, 4), (4, 4), (4, 2)])   # Sub-index 1
        ],
        2: [
            Polygon([(2, 2), (2, 4), (4, 4), (4, 2)])   # Sub-index 0
        ],
        3: [
            Polygon([(4, 4), (4, 6), (6, 6), (6, 4)])   # Sub-index 0
        ]
    }

    # Assign labels to object indices
    labels_ex, polygon_subindices_ex = assign_labels_to_object_indices(object_indices_ex, polygons_per_label_ex)

    print("Assigned Labels:", labels_ex)
    print("Polygon Sub-Indices:", polygon_subindices_ex)
    """
    # Ensure object_indices is a NumPy array for efficient processing
    assert isinstance(object_indices_list, list), ("object_indices must be a NumPy array or a list of tuples.")
    if len(object_indices_list) == 0:
        return [], []
    
    # Prepare a list of all polygons along with their corresponding labels and sub-indices
    all_polygons = []
    polygon_label_map = []
    polygon_label_subindex = []  # Indices of the polygons within their label key

    for label, polygons in polygons_per_label.items():
        all_polygons.extend(polygons)
        polygon_label_map.extend([label] * len(polygons))
        # Enumerate polygons to track their sub-indices within the label
        polygon_label_subindex.extend(list(range(len(polygons))))

    
    if not all_polygons:
        # If there are no polygons, return 0 labels and -1 sub-indices for all object indices
        num_objects = len(object_indices_list)
        return [0] * num_objects, [-1] * num_objects

    # Prepare polygons for faster spatial operations
    prepared_polygons = [shapely.prepared.prep(polygon) for polygon in all_polygons]

    # Build a spatial index for all polygons
    spatial_index = shapely.strtree.STRtree(all_polygons)

    assigned_labels = []
    polygon_label_subindex_found = []  # List to store the sub-index of the matching polygon

    for object_indicies in object_indices_list:
        label_found = 0  # Default label if no polygon contains the point
        subindex_found = -1  # Default sub-index if no polygon contains the point
        _running = True
        
        for obj_coord in object_indicies:
            point = shapely.geometry.Point(obj_coord)
            # Query the spatial index for possible containing polygons
            possible_polygons = spatial_index.query(point)

            for polygon_i in possible_polygons:
                # Retrieve the index of the polygon in all_polygons
                # polygon = all_polygons[polygon_i]
                if prepared_polygons[polygon_i].contains(point):
                    label_found = polygon_label_map[polygon_i]
                    subindex_found = polygon_label_subindex[polygon_i]
                    _running = False
                    break  # Stop after the first matching polygon is found
            if not _running:
                break

        assigned_labels.append(label_found)
        polygon_label_subindex_found.append(subindex_found)

    return assigned_labels, polygon_label_subindex_found






def semantic_to_polygons_rasterio(sem_image):
    """
    Convert a 2D semantic segmentation image into polygon boundaries using
    rasterio.features.shapes. This preserves exact pixel edges.
    
    Parameters
    ----------
    sem_image : np.ndarray
        A 2D integer array of shape (H, W) where each pixel is a label.
        E.g., 0 = background, 1 = object1, 2 = object2, etc.
    
    Returns
    -------
    polygons_dict : dict
        Dictionary of the form: { label_value: [Polygon, Polygon, ...], ... }
        Each label key maps to one or more Shapely Polygons representing
        the regions of that label.
    """
    polygons_dict = {}
    labels = np.unique(sem_image)

    for label in labels:
        # (Optional) Skip background label if you don't need it
        if label == 0:
            continue
        
        # Create a boolean mask for this label
        mask = (sem_image == label)
        
        # shapes() yields (geom, val) pairs for the "True" region of the mask
        #  - The first argument to shapes() is the raster (uint8 array),
        #    which in this case is just the binary mask for the label.
        #  - The `mask` parameter ensures we only polygonize the "True" region.
        results = rasterio.features.shapes(mask.astype(np.uint8), mask=mask, transform=rasterio.Affine(1, 0, 0, 0, 1, 0))
        
        # Convert each GeoJSON-like geometry dict to a Shapely shape
        region_polygons = []
        for geom, value in results:
            poly = shapely.geometry.shape(geom)
            if not poly.is_empty:
                region_polygons.append(poly)
        
        # Merge all polygon fragments into one or more polygons
        if region_polygons:
            merged = shapely.ops.unary_union(region_polygons)
            
            # merged could be a single Polygon or a MultiPolygon
            if merged.geom_type == 'MultiPolygon':
                polygons_dict[label] = list(merged.geoms)
            else:
                polygons_dict[label] = [merged]
        else:
            polygons_dict[label] = []

    return polygons_dict



def plot_polygons_over_image(sem_image, polygons_dict):
    plt.figure(figsize=(12, 12))
    plt.imshow(sem_image, cmap="gray")
    # plt.gca().invert_yaxis()  # uncomment if you want cartesian-like axes
    
    colors = ["red", "green", "blue", "yellow", "magenta"]
    seen_labels = set()
    for i, (label, polys) in enumerate(polygons_dict.items()):
        for poly in polys:
            x, y = poly.exterior.xy
            color = colors[i % len(colors)]
            plt.plot(x, y, color=color, linewidth=2, label=f"Label {label}" if label not in seen_labels else None)
            seen_labels.add(label)
            
            
            # If there are holes:
            for hole in poly.interiors:
                hx, hy = hole.coords.xy
                plt.plot(hx, hy, color=color, linewidth=2, linestyle="--")

    plt.title("Polygons Overlaid on Semantic Label Image")
    # Show legend (optional—this can get crowded if many labels)
    plt.legend(loc="best")
    plt.show()



def test_semantic_to_polygons_rasterio():
    # Example usage
    sem_image = np.array([
        [1, 1, 0, 0, 0],
        [1, 0, 0, 2, 2],
        [1, 1, 1, 2, 2],
        [0, 1, 1, 2, 2],
        [0, 0, 0, 0, 0],
    ], dtype=np.uint8)

    polygons_per_label = semantic_to_polygons_rasterio(sem_image)
    plot_polygons_over_image(sem_image, polygons_per_label)
    






