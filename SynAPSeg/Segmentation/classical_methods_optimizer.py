"""
scripts for optimizing classical segemntation methods (3d watershed) on a instance segmentation training dataset
for method comparison of accuracy.

"""

import numpy as np
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.metrics import variation_of_information
from itertools import product
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List

def generate_synthetic_3d_data(shape=(64, 64, 64), n_objects=10):
    """
    Generates a synthetic binary mask and corresponding ground truth labels 
    of touching spheres to simulate a segmentation task.
    """
    labels = np.zeros(shape, dtype=int)
    mask = np.zeros(shape, dtype=bool)
    
    rng = np.random.default_rng(42)
    
    for i in range(1, n_objects + 1):
        # Random center and radius
        center = rng.integers(10, shape[0]-10, size=3)
        radius = rng.integers(5, 12)
        
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        dist_from_center = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
        
        obj_mask = dist_from_center <= radius
        
        # Add to labels (handle overlap roughly by overwriting)
        labels[obj_mask] = i
        mask[obj_mask] = True
        
    return mask, labels

def calculate_average_iou(pred_labels: np.ndarray, gt_labels: np.ndarray) -> float:
    """
    Calculates the Average Intersection over Union (IoU) for instance segmentation.
    
    Strategy: For every Ground Truth (GT) object, find the Predicted object 
    that has the maximum overlap. Calculate IoU for that pair. Average over all GT objects.
    """
    unique_gt = np.unique(gt_labels)
    unique_gt = unique_gt[unique_gt != 0] # Exclude background
    
    if len(unique_gt) == 0:
        return 0.0

    ious = []
    
    for gt_id in unique_gt:
        gt_mask = (gt_labels == gt_id)
        gt_area = np.sum(gt_mask)
        
        # Find the label in prediction that overlaps most with this GT object
        # We look only at the area within the GT mask to save computation
        overlapping_preds = pred_labels[gt_mask]
        
        if overlapping_preds.size == 0:
            ious.append(0.0)
            continue
            
        # Get counts of predicted labels falling inside this GT object
        pred_ids, counts = np.unique(overlapping_preds, return_counts=True)
        
        # Remove background (0) from consideration if it overlaps
        valid_indices = pred_ids != 0
        if not np.any(valid_indices):
            ious.append(0.0)
            continue
            
        pred_ids = pred_ids[valid_indices]
        counts = counts[valid_indices]
        
        # Get best matching prediction ID
        best_match_idx = np.argmax(counts)
        best_pred_id = pred_ids[best_match_idx]
        intersection = counts[best_match_idx]
        
        # Calculate Union
        pred_mask = (pred_labels == best_pred_id)
        pred_area = np.sum(pred_mask)
        union = gt_area + pred_area - intersection
        
        iou = intersection / union if union > 0 else 0
        ious.append(iou)
        
    return np.mean(ious)

def run_3d_watershed(binary_mask: np.ndarray, sigma: float, min_distance: int) -> np.ndarray:
    """
    Performs 2D/3D watershed segmentation based on distance transform.
    
    Args:
        binary_mask: ZYX boolean array.
        sigma: Sigma for Gaussian smoothing of the distance map.
        min_distance: Minimum distance between peaks (markers).
        
    Returns:
        Labeled 3D numpy array.
    """
    # 1. Distance Transform
    distance = ndi.distance_transform_edt(binary_mask)

    # 2. Smooth the distance map to reduce noise (over-segmentation)
    # Sigma can be a float or a tuple (sigma_z, sigma_y, sigma_x) if anisotropic
    smoothed_distance = ndi.gaussian_filter(distance, sigma=sigma)

    # 3. Find peaks (local maxima) in the distance map
    # Ideally, each peak corresponds to the center of an object
    coords = peak_local_max(
        smoothed_distance, 
        min_distance=min_distance, 
        labels=binary_mask,
        exclude_border=False
    )

    # Create markers array
    markers = np.zeros_like(binary_mask, dtype=int)
    for i, coord in enumerate(coords):
        markers[tuple(coord)] = i + 1

    # Watershed 
    labels = watershed(
        image=-smoothed_distance, # negative because watershed fills "basins" (low points), but our objects are "peaks"
        markers=markers,
        mask=binary_mask,
        connectivity=binary_mask.ndim,
    )

    return labels

def optimize_watershed_params(
    binary_mask: np.ndarray, 
    gt_labels: np.ndarray, 
    param_grid: Dict[str, List]
) -> Tuple[Dict, float, np.ndarray]:
    """
    Grid search to optimize watershed parameters.
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))
    
    best_iou = -1.0
    best_params = {}
    best_prediction = np.zeros_like(binary_mask, dtype=np.int32)
    
    # print(f"Starting optimization with {len(combinations)} combinations...")
    # print(f"{'Sigma':<10} | {'Min Dist':<10} | {'IoU':<10}")
    # print("-" * 35)
    
    for combo in combinations:
        # Create a dictionary for current parameters
        params = dict(zip(keys, combo))
        
        try:
            # Run segmentation
            prediction = run_3d_watershed(
                binary_mask, 
                sigma=params['sigma'], 
                min_distance=params['min_distance']
            )
            prediction = uip.remove_small_objs(prediction, 5, binary_mask.ndim)
            
            # Evaluate
            iou = calculate_average_iou(prediction, gt_labels)
                        
            if iou > best_iou:
                best_iou = iou
                best_params = params
                best_prediction = prediction
                
        except Exception as e:
            print(f"Failed for params {params}: {e}")
            continue
            
    return best_params, best_iou, best_prediction

from typing import Callable

def get_best_result(options:list[Callable], test_img:np.ndarray, gt_labels:np.ndarray):
    
    best_iou = -1.0
    best_func = None
    best_prediction = None

    for func in options:    
        try:
            # Run segmentation
            prediction = func(test_img)
            prediction = uip.relabel(prediction)
            prediction = uip.remove_small_objs(prediction, 5, prediction.ndim)
                        
            # Evaluate
            iou = calculate_average_iou(prediction, gt_labels)
            
            print(f">{func} --> {iou:<10.4f}")
            
            if iou > best_iou:
                best_iou = iou
                best_func = func
                best_prediction = prediction
                
        except Exception as e:
            print(f"Failed for {func}: {e}")
            continue
    
    return best_func, best_iou, best_prediction

class SegMethod:
    def __init__(self, name, method:Callable, *args, **kwargs):
        self.name = name
        self.method = method
        self.args = args or ()
        self.kwargs = kwargs or {}
    
    def __str__(self):
        return self.name
    def __repr__(self):
        return f"{self.__class__.__name__}.{self}"
    
    def __call__(self, arr:np.ndarray):
        return self.method(arr, **self.kwargs)

class ThreshSegMethod(SegMethod):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thold = 0.0
    
    def __call__(self, arr:np.ndarray):
        self.thold = self.method(arr, **self.kwargs)
        return arr>self.thold

def get_auto_thresh_methods():
    
    from skimage import filters

    # Global algorithms.
    thresh = lambda x: x
    methods = {
        'Isodata': thresh(filters.threshold_isodata),
        'Li': thresh(filters.threshold_li),
        'Mean': thresh(filters.threshold_mean),
        'Otsu': thresh(filters.threshold_otsu),
        'Triangle': thresh(filters.threshold_triangle),
        'Yen': thresh(filters.threshold_yen),
    }

    return [
        ThreshSegMethod(name=k, method=v) for k,v in methods.items()
    ]


import os
import sys
from pathlib import Path
from SynAPSeg.Train.benchmark import get_benchmark_dataset_as_testset
from SynAPSeg.config import constants
from SynAPSeg.utils import utils_image_processing as uip
from SynAPSeg.utils import utils_plotting as up
from SynAPSeg.utils import utils_general as ug
from SynAPSeg.utils import utils_ML as uML

# verify_and_set_env_dirs()


def run_optimize_classic_seg(img, gt_labels, thresh_method=None, param_grid=None):
    """ segment an image using classical threshold method watershed
            use gt labels to optimize 3d watershed params via maximize iou 
    """
    from skimage import filters
    thresh_method = thresh_method or {'yen':filters.threshold_yen}
    k = list(thresh_method.keys())[0]
    thresh_method = ThreshSegMethod(name=k, method=thresh_method[k])
    
    
    # generate mask 
    mask = thresh_method(img)
    mask = uip.relabel(mask)
    mask = uip.remove_small_objs(mask, 5, 3)
    mask = mask > 0
    
    # 3. Run Optimization
    param_grid = param_grid or {
        'sigma': [0.5, 1.0, 1.5, 2.0],
        'min_distance': [3, 5, 8, 12, 15]
    }
    best_params, best_iou, best_pred = optimize_watershed_params(mask, gt_labels, param_grid)

    return best_params, best_pred


def batch_classic_seg(imgs, gt_labels, **kwargs):
    """ returns list of best params, list of predictions """

    res = [run_optimize_classic_seg(im, gt, **kwargs) for im, gt in zip(imgs, gt_labels)]    
    best_params = [r[0] for r in res]
    Y = [r[1] for r in res]
    return best_params, Y


if bool(0): # run classic seg w/ single threshold and optimized 3d watershed over 3d training set
    from Train.stardist_3d_3_evaluate_2025119 import get_data

    TRAIN_DATA_BASE_DIR = r"J:\__compiled_training_data__\3d_synapses\patchified"
    DATA_DIR_NAME = '2025_1125_32x128x128'
    X, Y = get_data(TRAIN_DATA_BASE_DIR, DATA_DIR_NAME)

    best_params, preds = batch_classic_seg(testX, testY) # best params were sigma=0.5, min_dist=3
    iou_bin = [Bench.compute_iou(pred>0, gt>0) for pred, gt in zip(preds, testY)]
    print(f"aggregate iou binary over test set: {np.mean(np.array(iou_bin))}")  # res was 0.40

    up.plot_image_grid([uip.mip(a) for a in X[9:18]], n_cols=3)
    up.plot_image_grid([uip.mip(a) for a in preds[9:18]], n_cols=3)


if __name__ == "__main__": #def main():
    
    # test a single image
    ####################################################################################

    # 1. Setup Data
    if bool(0): # synthetic - test
        print("Generating synthetic 3D data (Blobs)...")
        mask, gt_labels = generate_synthetic_3d_data(shape=(60, 100, 100), n_objects=8)
    else:
        # load benchmark dataset 
        X, gt_labels = get_benchmark_dataset_as_testset('syn3d_0')
        img, gt_labels = X[0], gt_labels[0]

        # generate binary mask through thresholding - try different, get best
        trymethods = get_auto_thresh_methods()
        best_func, best_thresh_iou, mask = get_best_result(trymethods, img, gt_labels)
        mask = mask>0

    # 2. Define Parameter Search Space
    # sigma: Controls smoothness of the "landscape". Higher = less splitting.
    # min_distance: Controls how close two object centers can be. Higher = merges objects.
    param_grid = {
        'sigma': [0.5, 1.0, 1.5, 2.0],
        'min_distance': [3, 5, 8, 12, 15]
    }
    
    # 3. Run Optimization
    best_params, best_iou, best_pred = optimize_watershed_params(mask, gt_labels, param_grid)

    n_objs_gt = len(uip.unique_nonzero(gt_labels))
    n_objs_pred = len(uip.unique_nonzero(best_pred))
    
    print("\n" + "="*40)
    print(f"Optimization Complete.")
    print(f"Best IOU: {best_iou:.4f}")
    print(f"Best Parameters: {best_params}")
    print("="*40)
    
    # 4. Visualize Results (Center Slice of Z-axis)
    z_slice = mask.shape[0] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img[z_slice], cmap='gray')
    axes[0].set_title(f"Input image (Z={z_slice})")
    
    axes[1].imshow(gt_labels[z_slice], cmap='nipy_spectral', interpolation='nearest')
    axes[1].set_title(f"Ground Truth Labels\n(n objs: {n_objs_gt})")
    
    used_thresh = str(round(best_func.thold,2))
    axes[2].imshow(best_pred[z_slice], cmap='nipy_spectral', interpolation='nearest')
    axes[2].set_title(f"{best_func} filter + 3D watershed\n(n objs: {n_objs_pred}, thresh={used_thresh}, s={best_params['sigma']}, d={best_params['min_distance']})")
    for ax in axes.flatten():
        ax.axis('off')
    plt.tight_layout()
    up.save_fig('testing_classical_segmentation_methods_syn3d.svg')
    plt.show()

    # plot whole stack
    ##############################################################
    up.plot_image_grid(
        [uip.mip(a) for a in [img, gt_labels, best_pred]],
        cmap='grey',
        n_cols=3
    )

    # mean average prec
    ##############################################################
    from cellpose import metrics
    ap, tp, fp, fn = metrics.average_precision([best_pred], [gt_labels])
    from Train.benchmark import compute_ap_at_iou
    mAP = compute_ap_at_iou([tuple(('id1', best_pred, 1.0))], [tuple(('id1', gt_labels))], 0.5)
    from Train.benchmark import compute_iou
    IOU = compute_iou(best_pred>0, gt_labels>0)
    
    print('mAP, tp,fp,fn', ap, tp,fp,fn)
    print(f'avg. IOU: {best_iou}\nbinary IOU: {IOU}')
