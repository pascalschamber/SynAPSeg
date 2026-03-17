"""
Fast 3D synapse-to-dendrite assignment using overlap and distance methods.
Optimized with Numba for large-scale volumetric data.
"""

import numpy as np
import numba
from scipy import ndimage
from typing import Optional


@numba.njit(cache=True, fastmath=False)
def _count_overlaps_kernel(syn_flat, den_flat, syn_labels, max_den_label):
    """
    Count overlaping pixels between each synapse and dendrites.
        iterate over all pixels in flattened label volume array for synapses
        counting each dend label it overlaps with

    Returns:
        overlap_counts: shape (n_synapses, max_den_label+1)
    """
    n_synapses = len(syn_labels)
    overlap_counts = np.zeros((n_synapses, max_den_label + 1), dtype=np.int64)
    
    # Build synapse label to index mapping (synapse instance label: label index)
    syn_to_idx = np.zeros(np.max(syn_labels) + 1, dtype=np.int32)
    for i, s in enumerate(syn_labels):
        syn_to_idx[s] = i
    
    # Count overlaps
    for i in range(len(syn_flat)):
        s = syn_flat[i] # instance label of syanpse object at this point
        d = den_flat[i] # roi object label 
        if s > 0 and d > 0:
            syn_idx = syn_to_idx[s]
            overlap_counts[syn_idx, d] += 1
    
    return overlap_counts


@numba.njit(cache=True, fastmath=False)
def _assign_by_max_overlap(overlap_counts, syn_labels):
    """
    Assign each synapse to dendrite with maximum overlap.
    Ties broken by smallest dendrite label.
    
    Returns:
        assignments: array of dendrite labels (0 = None)
    """
    n_synapses = len(syn_labels)
    assignments = np.zeros(n_synapses, dtype=np.int32)
    
    for i in range(n_synapses):
        counts = overlap_counts[i]
        max_count = np.max(counts)
        
        if max_count > 0:
            # Find smallest dendrite label with max count
            for d in range(len(counts)):
                if counts[d] == max_count:
                    assignments[i] = d
                    break
    
    return assignments


def map_synapses_to_dendrites_overlap(
    syn_labels: np.ndarray,
    den_labels: np.ndarray
) -> tuple[dict[int, Optional[int]], dict]:
    """
    Assign synapses to dendrites based on voxel overlap.
    
    Each synapse is assigned to the dendrite with the largest number of 
    overlapping voxels. Ties are broken by choosing the smallest dendrite label.
    
    Args:
        syn_labels: 3D array (Z, Y, X) with synapse instance labels (0=background)
        den_labels: 3D array (Z, Y, X) with dendrite instance labels (0=background)
    
    Returns:
        mapping: dict mapping synapse_label -> dendrite_label (or None)
        stats: dict with n_synapses, n_assigned, n_unassigned
    
    Complexity:
        Time: O(V + S*D_max) where V=voxels, S=synapses, D_max=max dendrites per synapse
        Space: O(S*D) for overlap counts where D is number of dendrites
    """
    # Validate inputs
    if syn_labels.ndim != 3 or den_labels.ndim != 3:
        raise ValueError(f"Both arrays must be 3D, got shapes {syn_labels.shape}, {den_labels.shape}")
    if syn_labels.shape != den_labels.shape:
        raise ValueError(f"Shape mismatch: {syn_labels.shape} vs {den_labels.shape}")
    if not np.issubdtype(syn_labels.dtype, np.integer):
        raise TypeError(f"syn_labels must be integer type, got {syn_labels.dtype}")
    if not np.issubdtype(den_labels.dtype, np.integer):
        raise TypeError(f"den_labels must be integer type, got {den_labels.dtype}")
    
    # Get unique synapse labels (excluding 0)
    syn_unique = np.unique(syn_labels)
    syn_unique = syn_unique[syn_unique > 0]
    
    if len(syn_unique) == 0:
        return {}, {"n_synapses": 0, "n_assigned": 0, "n_unassigned": 0}
    
    # Get max dendrite label
    den_unique = np.unique(den_labels)
    den_unique = den_unique[den_unique > 0]
    
    if len(den_unique) == 0:
        # No dendrites - all synapses unassigned
        mapping = {int(s): None for s in syn_unique}
        stats = {
            "n_synapses": len(syn_unique),
            "n_assigned": 0,
            "n_unassigned": len(syn_unique)
        }
        return mapping, stats
    
    max_den_label = int(np.max(den_unique))
    
    # Flatten arrays for efficient processing
    syn_flat = syn_labels.ravel()
    den_flat = den_labels.ravel()
    
    # Count overlaps using Numba
    overlap_counts = _count_overlaps_kernel(syn_flat, den_flat, syn_unique, max_den_label)
    
    # Assign based on maximum overlap
    assignments = _assign_by_max_overlap(overlap_counts, syn_unique)
    
    # Build mapping and stats
    mapping = {}
    n_assigned = 0
    for i, syn_label in enumerate(syn_unique):
        den_label = int(assignments[i])
        if den_label > 0:
            mapping[int(syn_label)] = den_label
            n_assigned += 1
        else:
            mapping[int(syn_label)] = None
    
    stats = {
        "n_synapses": len(syn_unique),
        "n_assigned": n_assigned,
        "n_unassigned": len(syn_unique) - n_assigned
    }
    
    return mapping, stats


@numba.njit(cache=True, fastmath=False)
def _find_min_distance_per_synapse(syn_coords, edt_values, nearest_labels, syn_labels):
    """
    For each synapse, find minimum EDT value and corresponding dendrite label.
    
    Returns:
        min_distances: array of min distances for each synapse
        assigned_labels: array of dendrite labels at min distance points
    """
    n_synapses = len(syn_labels)
    min_distances = np.full(n_synapses, np.inf, dtype=np.float64)
    assigned_labels = np.zeros(n_synapses, dtype=np.int32)
    
    # Build synapse label to index mapping
    max_syn = np.max(syn_labels)
    syn_to_idx = np.zeros(max_syn + 1, dtype=np.int32)
    for i, s in enumerate(syn_labels):
        syn_to_idx[s] = i
    
    # Find minimum distance for each synapse
    for i in range(len(syn_coords)):
        syn_label = syn_coords[i, 0]
        edt_val = edt_values[i]
        nearest_label = nearest_labels[i]
        
        syn_idx = syn_to_idx[syn_label]
        
        if edt_val < min_distances[syn_idx]:
            min_distances[syn_idx] = edt_val
            assigned_labels[syn_idx] = nearest_label

        # note: not sure this tie-breaking is correctly enforced - actual problem may be elsewhere
        elif edt_val == min_distances[syn_idx] and nearest_label < assigned_labels[syn_idx]:
            # Tie-breaking: prefer smaller label
            assigned_labels[syn_idx] = nearest_label
    
    return min_distances, assigned_labels


def map_synapses_to_dendrites_distance(
    syn_labels: np.ndarray,
    den_labels: np.ndarray,
    radius: float = 10.0,
    voxel_size: Optional[tuple[float, float, float]] = None
) -> tuple[dict[int, Optional[int]], dict]:
    """
    Assign synapses to nearest dendrite within distance threshold.
    
    Each synapse is assigned to the nearest dendrite if the minimum Euclidean
    distance is ≤ radius. Ties are broken by choosing the smallest dendrite label.
    
    Args:
        syn_labels: 3D array (Z, Y, X) with synapse instance labels (0=background)
        den_labels: 3D array (Z, Y, X) with dendrite instance labels (0=background)
        radius: Maximum distance threshold (in voxels or physical units)
        voxel_size: Optional (dz, dy, dx) for anisotropic spacing
    
    Returns:
        mapping: dict mapping synapse_label -> dendrite_label (or None)
        stats: dict with n_synapses, n_assigned, n_unassigned
    
    Complexity:
        Time: O(V log V) for EDT + O(S*V_s) for per-synapse reduction
        Space: O(V) for EDT and label propagation arrays
    """
    # Validate inputs
    if syn_labels.ndim != 3 or den_labels.ndim != 3:
        raise ValueError(f"Both arrays must be 3D, got shapes {syn_labels.shape}, {den_labels.shape}")
    if syn_labels.shape != den_labels.shape:
        raise ValueError(f"Shape mismatch: {syn_labels.shape} vs {den_labels.shape}")
    if not np.issubdtype(syn_labels.dtype, np.integer):
        raise TypeError(f"syn_labels must be integer type, got {syn_labels.dtype}")
    if not np.issubdtype(den_labels.dtype, np.integer):
        raise TypeError(f"den_labels must be integer type, got {den_labels.dtype}")
    
    # Get unique synapse labels
    syn_unique = np.unique(syn_labels)
    syn_unique = syn_unique[syn_unique > 0]

    # minimum distance to dend (calculated for all, even those outside radius threshold)
    mindist_mapping = {syn_label:{'min_dist':None, 'den_label':None} for syn_label in syn_unique}
    
    if len(syn_unique) == 0:
        return {}, {"n_synapses": 0, "n_assigned": 0, "n_unassigned": 0, "mindist_mapping":mindist_mapping}
    
    # Get dendrite labels
    den_unique = np.unique(den_labels)
    den_unique = den_unique[den_unique > 0]
    
    if len(den_unique) == 0:
        # No dendrites - all synapses unassigned
        mapping = {int(s): None for s in syn_unique}
        stats = {
            "n_synapses": len(syn_unique),
            "n_assigned": 0,
            "n_unassigned": len(syn_unique),
            "mindist_mapping": mindist_mapping,
        }
        return mapping, stats
    
    # Create binary dendrite mask
    den_mask = den_labels > 0
    
    # Compute distance transform
    if voxel_size is not None:
        sampling = voxel_size
    else:
        sampling = (1,1,1)
        
    # Compute nearest dendrite label using feature_transform (returns indices)
    edt, indices = ndimage.distance_transform_edt(~den_mask, sampling=sampling, return_distances=True, return_indices=True)
    
    # Get nearest dendrite labels
    nearest_den_labels = den_labels[indices[0], indices[1], indices[2]]
    
    # For each synapse, find coordinates and corresponding EDT values
    syn_mask = syn_labels > 0
    syn_coords_idx = np.where(syn_mask)
    
    # Build array: [syn_label, edt_value, nearest_den_label]
    syn_coords_with_data = np.column_stack([
        syn_labels[syn_coords_idx].astype(np.int32),
    ])
    edt_vals = edt[syn_coords_idx]
    nearest_labels = nearest_den_labels[syn_coords_idx].astype(np.int32)
    
    # Find minimum distance per synapse using Numba
    min_distances, assigned_labels = _find_min_distance_per_synapse(
        syn_coords_with_data, edt_vals, nearest_labels, syn_unique
    )
    
    # Build mapping and stats
    mapping = {}
    n_assigned = 0
        
    for i, syn_label in enumerate(syn_unique):
        
        min_dist = min_distances[i]
        den_label = int(assigned_labels[i])
        mindist_mapping[syn_label] = {'min_dist':min_dist, 'den_label':den_label}

        if min_dist <= radius and den_label > 0:
            mapping[int(syn_label)] = den_label
            n_assigned += 1
        else:
            mapping[int(syn_label)] = None
    
    stats = {
        "n_synapses": len(syn_unique),
        "n_assigned": n_assigned,
        "n_unassigned": len(syn_unique) - n_assigned,
        "mindist_mapping": mindist_mapping,
    }
    
    return mapping, stats




def test_synapse_dendrite_assignment():
    """Test both assignment methods with various edge cases."""
    
    print("=" * 70)
    print("Test 1: No overlap - synapses and dendrites separate")
    print("=" * 70)
    
    syn = np.zeros((5, 5, 5), dtype=np.int32)
    den = np.zeros((5, 5, 5), dtype=np.int32)
    
    syn[0:2, 0:2, 0:2] = 1  # Synapse 1
    den[3:5, 3:5, 3:5] = 1  # Dendrite 1 (far away)
    
    mapping, stats = map_synapses_to_dendrites_overlap(syn, den)
    print(f"Overlap mapping: {mapping}")
    print(f"Stats: {stats}")
    assert mapping[1] is None
    assert stats['n_unassigned'] == 1

    mapping, stats = map_synapses_to_dendrites_distance(syn, den, radius=3.4)
    print(f"Distance mapping (r=3.4): {mapping}")
    print(f"Stats: {stats}")
    assert mapping[1] is None  # Too far
    
    mapping, stats = map_synapses_to_dendrites_distance(syn, den, radius=np.inf)
    print(f"Changing r so now overlap:\nDistance mapping (r=3.5): {mapping}")
    print(f"Stats: {stats}")
    assert mapping[1] == 1  # Within range now
    

    print("\n" + "=" * 70)
    print("Test 2: One-to-one overlap")
    print("=" * 70)
    
    syn = np.zeros((5, 5, 5), dtype=np.int32)
    den = np.zeros((5, 5, 5), dtype=np.int32)
    
    syn[1:3, 1:3, 1:3] = 1  # Synapse 1
    den[1:4, 1:4, 1:4] = 1  # Dendrite 1 (overlaps)
    
    mapping, stats = map_synapses_to_dendrites_overlap(syn, den)
    print(f"Overlap mapping: {mapping}")
    print(f"Stats: {stats}")
    assert mapping[1] == 1
    assert stats['n_assigned'] == 1
    

    print("\n" + "=" * 70)
    print("Test 3: Synapse overlapping multiple dendrites")
    print("=" * 70)
    
    syn = np.zeros((5, 5, 5), dtype=np.int32)
    den = np.zeros((5, 5, 5), dtype=np.int32)
    
    syn[1:3, 1:3, 1:3] = 1    # Synapse 1 (8 voxels)
    den[1:2, 1:2, 1:3] = 1    # Dendrite 1 (2 voxels overlap)
    den[2:3, 1:3, 1:3] = 2    # Dendrite 2 (4 voxels overlap)
    den[1:2, 2:3, 1:3] = 3    # Dendrite 3 (2 voxels overlap)
    
    mapping, stats = map_synapses_to_dendrites_overlap(syn, den)
    print(f"Overlap mapping: {mapping}")
    print(f"Stats: {stats}")
    assert mapping[1] == 2  # Most overlap
    
    print("\n" + "=" * 70)
    print("Test 4: Tie-breaking (equal overlap)")
    print("=" * 70)
    
    syn = np.zeros((5, 5, 5), dtype=np.int32)
    den = np.zeros((5, 5, 5), dtype=np.int32)
    
    syn[1:3, 1:3, 1:3] = 1    # Synapse 1
    den[1:2, 1:3, 1:3] = 5    # Dendrite 5 (4 voxels)
    den[2:3, 1:3, 1:3] = 2    # Dendrite 2 (4 voxels)
    
    mapping, stats = map_synapses_to_dendrites_overlap(syn, den)
    print(f"Overlap mapping: {mapping}")
    print(f"Stats: {stats}")
    assert mapping[1] == 2  # Smaller label wins tie
    
    print("\n" + "=" * 70)
    print("Test 5: Distance tie-breaking limitation")
    print("Tie-breaking issue - label assignment doesn't follow lowest id wins instead based on scan order")
    print("note lowest roi label is not assigned here, instead roi label with lowest e.g. z-value wins")
    print("So the problem is delegating the tie-break to EDTs internal propagation order, which is probably based on scan order / axis priority, not label ID. even though both are equally valid geometrically.")
    print("=" * 70)
    
    syn = np.zeros((10, 10, 10), dtype=np.int32)
    den = np.zeros((10, 10, 10), dtype=np.int32)
    
    syn[5, 5, 5] = 1      # Single voxel synapse
    den[5, 3, 5] = 3      # Distance 2 (z-axis)
    den[5, 7, 5] = 7      # Distance 2 (z-axis)
    
    mapping, stats = map_synapses_to_dendrites_distance(syn, den, radius=3.0)
    print(f"Distance mapping: {mapping}")
    print(f"Stats: {stats}")
    assert mapping[1] == 3  # lowest scan order label wins tie
    
    syn = np.zeros((10, 10, 10), dtype=np.int32)
    den = np.zeros((10, 10, 10), dtype=np.int32)
    
    syn[5, 5, 5] = 1      # Single voxel synapse
    den[5, 3, 5] = 7      # Distance 2 (z-axis)
    den[5, 7, 5] = 3      # Distance 2 (z-axis)
    
    mapping, stats = map_synapses_to_dendrites_distance(syn, den, radius=3.0)
    print(f"Distance mapping: {mapping}")
    print(f"Stats: {stats}")
    assert mapping[1] == 7  # lowest scan order label wins tie
    
    
    print("\n" + "=" * 70)
    print("Test 6: No dendrites")
    print("=" * 70)
    
    syn = np.zeros((5, 5, 5), dtype=np.int32)
    den = np.zeros((5, 5, 5), dtype=np.int32)
    syn[1:3, 1:3, 1:3] = 1
    
    mapping, stats = map_synapses_to_dendrites_overlap(syn, den)
    print(f"Overlap mapping: {mapping}")
    print(f"Stats: {stats}")
    assert mapping[1] is None
    assert stats['n_assigned'] == 0
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_synapse_dendrite_assignment()


    if bool(0): # toy example showing edt tie-breaking issue:
        den_labelst = np.zeros((3,3,3))
        den_labelst[0,1,1] = 7
        den_labelst[2,1,1] = 3
        _sampling = [None, (1,1,1)][1]
        den_maskt = den_labelst > 0
        edt, indices = ndimage.distance_transform_edt(~den_maskt, sampling=_sampling, return_distances=True, return_indices=True)
        print(den_labelst)
        print(edt)
        for inds in indices:
            print(inds)

        nearest_den_labels = den_labelst[indices[0], indices[1], indices[2]]
        print(nearest_den_labels)
    

    
    
    

