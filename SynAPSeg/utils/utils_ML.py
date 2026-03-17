import numpy as np
from copy import deepcopy
import os
import sys

try: 
    import volumentations as volaug
except Exception as e:
    print(f"-- in utils_ML -- failed to import volumentations -- 3d augmenter not available \n{e}")
    volaug = None


def clear_gpu_memory(K=None, torch=None):
    """ clear gpu memory """
    try:
        if K is None:
            import tensorflow.keras.backend as K
        K.clear_session()
    except:
        pass
    
    if torch is not None:
        try:
            torch.cuda.empty_cache()
        except:
            pass


def cuda_mem_alloc(prefix='', device_idx=0, p=False):
    """ get GPU memory usage (not just pytorch usage) """
    try:
        import torch
    except:
        print('torch not imported')
        return None
        
    value = torch.cuda.device_memory_used(device_idx)
    formatted_value = "{:.2e}".format(value) # Format with 2 decimal places
    res = f"{prefix}{formatted_value}"
    if p:
        print(res)
    return res


def check_gpu():
    """
    Checks if TensorFlow can access GPU.
    
    Returns:
        bool: True if a GPU is available for TensorFlow, False otherwise.

    Note:
        can check from CLI with the following command:
            python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
            python -c "import torch; print(torch.cuda.is_available())"
    """
    try:
        import tensorflow as tf
    except Exception as e:
        print(f"-- in utils_ML -- failed to import tensorflow -- gpu check not available \n{e}")
        return False

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            print(gpus)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            return True
        except RuntimeError as e:
            print(e)
            return False
    else:
        return False


class Augmentor:
    """ wrapper around Volumentations 3D, so augmentation can be a callable e.g. usable by stardist.model.train"""

    def __init__(self, augmenter, img_dtype=np.float32, lbl_dtype=np.int32, enforce_dtypes=True, enforce_unique_labels = True):
        """
        Initialize the Augmentor class.

        Parameters:
        patch_size: tuple
            The desired patch size for resizing the images.
        augmenter: volaug.Compose
            A volumentations Compose object containing the augmentations to apply.
        img_dtype: numpy.dtype
            The desired data type for the images.
        lbl_dtype: numpy.dtype
            The desired data type for the labels.
        enforce_dtypes: bool
            Whether to enforce the specified data types for the images and labels whenever augmentations are applied.
        """
        self.augmenter = augmenter
        self.img_dtype = img_dtype
        self.lbl_dtype = lbl_dtype
        self.enforce_dtypes = enforce_dtypes
        self.enforce_unique_labels = enforce_unique_labels # always relabel 
    
    def get_config(self) -> list[dict]:
        """ 
        returns a list[dict] containing params used to init each transform, 
            e.g. for logging params used in a run.
        """
        return [{'name':t.__class__.__name__, 'params':t.__dict__} for t in self.augmenter.transforms]

    def __call__(self, img, lbl):
        """
        allows object to function as callable
        """
        return self.apply_augs(self.augmenter, img, lbl)
    
    def apply_augs(self, augcomp, img, lbl):
        """
        Apply the augmentations to the given image and label.

        Args:
            augcomp (volaug.Compose): The Compose object containing the augmentations to apply.
            img (numpy.ndarray): The input 3D image.
            lbl (numpy.ndarray): The corresponding label or mask.

        Returns:
            tuple: A tuple (img_aug, lbl_aug) containing the augmented image and label.
        """
        from SynAPSeg.utils.utils_image_processing import relabel

        aug_data = augcomp(image=img, mask=lbl)
        img_aug, lbl_aug = aug_data['image'], aug_data['mask']
        if self.enforce_unique_labels:
            lbl_aug = relabel(lbl_aug.astype(self.lbl_dtype), connectivity=lbl_aug.ndim)
        if self.enforce_dtypes:
            img_aug, lbl_aug = img_aug.astype(self.img_dtype), lbl_aug.astype(self.lbl_dtype)
        return img_aug, lbl_aug
    
    
    def test_individual_augmentations(self, img, lbl):
        """
        Test each augmentation in the Compose object individually.

        Args:
            img (numpy.ndarray): The input 3D image.
            lbl (numpy.ndarray): The corresponding label or mask.

        Returns:
            dict: A dictionary where each key is the augmentation name and each value is the result of applying that augmentation.
        """
        results = {}
        first_tform = self.augmenter.transforms[0] # this is dtype conversion
        for t_i, transform in enumerate(self.augmenter.transforms[1:]):
            
            # Wrap the transform in a single augmentation Compose for testing
            t = deepcopy(transform)
            t.always_apply = True
            
            single_transform = volaug.Compose([first_tform, t], p=1.0)
            img_aug, lbl_aug = self.apply_augs(single_transform, img, lbl)
            # aug_data = single_transform(image=img, mask=lbl)
            # img_aug, lbl_aug = aug_data['image'], aug_data['mask']
            results[f"{t_i} - {transform.__class__.__name__}"] = (img_aug, lbl_aug)
        return results
    
    def plot_test_aug_results(self, test_aug_res, img, lbl):
        """plot results from test_individual_augmentations"""
        from SynAPSeg.utils import utils_plotting as up
        from SynAPSeg.utils.utils_image_processing import pai, mip

        titles, imgs = [f'input img ({pai(img, asstr=True)})', f'input lbl ({pai(lbl, asstr=True)})'], [img, lbl]
        for k, res in test_aug_res.items():
            print(k, len(res))
            aimg, albl = res
            img_stats, lbl_stats = pai(aimg, asstr=True), pai(albl, asstr=True)
            titles.extend([f"{k} img ({img_stats})", f"{k} lbl ({lbl_stats})"])
            imgs.extend(list(res))
        return up.plot_image_grid([mip(x) for x in imgs], titles=titles, n_cols=4)

    def test_iter_apply(self, img, lbl, n=10):
        results = {}
        for i in range(n):
            results[i] = self(img, lbl)
        return results
        