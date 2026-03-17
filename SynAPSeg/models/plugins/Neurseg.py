import os
import sys
from pathlib import Path
from copy import deepcopy
import numpy as np
import gc
from typing import Tuple


from SynAPSeg.models.base import SegmentationModel

__plugin_group__ = "model"
__plugin__ = "NeursegModel"
__parameters__ = 'Neurseg.yaml'

class NeursegModel(SegmentationModel):
            
    def load_model(self, model_path, **load_model_kwargs):
        import tensorflow as tf
        import segmentation_models as sm
        from SynAPSeg.utils.utils_general import popget
        self.n_classes = popget(load_model_kwargs, 'n_classes', 1)
        self.backbone = popget(load_model_kwargs, 'backbone', None)
        
        custom_objects = {
            'dice_loss': sm.losses.DiceLoss(),
            'binary_focal_loss': sm.losses.binary_focal_loss,
            'binary_focal_loss_plus_jaccard_loss':sm.losses.binary_focal_jaccard_loss,
            'binary_focal_dice_loss':sm.losses.binary_focal_dice_loss,
            'iou_score': sm.metrics.iou_score, 'f1-score': sm.metrics.f1_score, 'precision': sm.metrics.precision, 'recall': sm.metrics.recall, 
            'n_classes': self.n_classes, 'backbone': self.backbone,
        }
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        
    def preprocess(self, input_array, **preprocessing_kwargs):
        """
        Custom preprocess implementation for this model.
        Normalizes the input to the range (0,1).
        """
        from SynAPSeg.utils.utils_image_processing import normalize_01
        
        return normalize_01(input_array).astype('float32') 
    
    def _to_patches(self, input_array, preprocess_input_fxn):
        """
        convert input array to patches compatible with model's expected input
        """
        import patchify
        from SynAPSeg.utils.utils_image_processing import pad_image_to_multiple
        # TODO can remove patchify dependency if we modify uip.coord_patch_iterator to handle n-dim patch coords
        # though only 2d is implemented here atm, would also need to implement unpatchify
        # also, patchify.patchify is just a wrapper for view_as_windows from skimage/util/shape.py

        og_shape = input_array.shape                                                                                # example of shape processing
        if self.debug: print(f"predict input_array shape: {og_shape}")                                              # np.zeros((1028, 1028))
        image = pad_image_to_multiple(input_array, self.pred_patch_shape[0])                                        # (1280, 1280)
        patches = patchify.patchify(image, list(self.pred_patch_shape), step=self.patchify_stepsize)                # (5, 5, 256, 256)
        flat_patches = patches.reshape(-1, *patches.shape[-2:])                                                      # (25, 256, 256)
        flat_preproc = preprocess_input_fxn(np.repeat(np.expand_dims(flat_patches, -1), 3, axis=-1)).astype('float32')  # (25, 256, 256, 3)
        self.patches_shape = patches.shape # store for unpatching
        return flat_preproc

    def predict(self, input_array, **predict_kwargs):        
        import tensorflow as tf
        import segmentation_models as sm
        import tensorflow.keras.backend as K
        import patchify
        from SynAPSeg.utils.utils_general import popget
        
        kwargs = deepcopy(predict_kwargs)
        self.BinPredThresh = popget(kwargs, 'BinPredThresh', 0.5)
        self.pred_patch_shape: Tuple[int, int] = popget(kwargs, 'patch_shape', (256,256))
        self.patchify_stepsize = self.pred_patch_shape[0] if self.pred_patch_shape else None
        preprocess_input = sm.get_preprocessing(self.backbone)
        
        # predict over patches
        og_shape = input_array.shape
        pred = self.model.predict(self._to_patches(input_array, preprocess_input))      

        pr_pred_flat = np.argmax(pred, axis=-1) if self.n_classes > 1 else (pred[...,0]>self.BinPredThresh).astype('int32')
        unflatten_pred = pr_pred_flat.reshape(*self.patches_shape[:2], *pr_pred_flat.shape[1:])
        unpatched = patchify.unpatchify(unflatten_pred, np.array(self.patches_shape[:-2]) * np.array(self.patches_shape[-2:])).astype('int32')
        unpadded = unpatched[:og_shape[0], :og_shape[1]]
        
        K.clear_session()
        gc.collect()

        return unpadded
    
    def postprocess(self, input_array: np.ndarray, **postprocessing_kwargs):
        """
        Default implementation of preprocess, returns the input. Subclasses should override this if they need custom behavior.
        """
        remove_small_objs_size = postprocessing_kwargs.get('remove_small_objs_size', None)
        
        if remove_small_objs_size is not None:
            from SynAPSeg.utils.utils_image_processing import remove_small_objs, relabel
            cleaned_img = remove_small_objs(relabel(input_array > 0), min_size=remove_small_objs_size, connectivity=2)
            return np.where(cleaned_img > 0, input_array, 0)
        return input_array


