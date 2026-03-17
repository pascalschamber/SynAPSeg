import os
import sys
from pathlib import Path
import numpy as np
import importlib
from inspect import getdoc

from SynAPSeg.models.base import SegmentationModel
import gc

__plugin_group__ = "model"
__plugin__ = "StardistModel"
__parameters__ = 'Stardist.yaml'
__append_docs__ = {
    'predict': 'stardist.models.base.StarDistBase.predict_instances',
    'preprocess': 'csbdeep.utils.normalize'
}

class StardistModel(SegmentationModel):
    def post_init(self):
        self._append_docs(__append_docs__)

    def load_model(self, model_path, **load_model_kwargs):
        weights_filename = load_model_kwargs.get('weights_filename') # default load weights_last.h5, options: 'weights_last.h5'

        from stardist.models import StarDist2D, StarDist3D
        model_dir, model_name = str(Path(model_path).parent), Path(model_path).name

        assert len(self.in_dims_model) in [2,3], f"in_dims_model must be of length 2 or 3 but got: {self.in_dims_model}"
        modeltype = StarDist2D if len(self.in_dims_model)==2 else StarDist3D
        model = modeltype(None, name=model_name, basedir=model_dir)
        
        if weights_filename and (weights_filename != 'weights_best.h5'): # these are loaded by default
            print("Loading network weights from '%s'." % weights_filename)
            model.load_weights(name=weights_filename)
        
        return model
        
    def preprocess(self, input_array, **preprocessing_kwargs):
        """
        Normalizes the input to the range (0,1).
            norm = preprocessing_kwargs.get('norm', (1, 99.8))
            clip = preprocessing_kwargs.get('clip', False)
        """
        from csbdeep import utils
        norm = preprocessing_kwargs.get('norm', (1, 99.8))
        clip = preprocessing_kwargs.get('clip', False)
        
        if norm is None:
            return input_array
                
        return utils.normalize(input_array, pmin=norm[0], pmax=norm[1], clip=clip)
    
    def predict(self, input_array, **predict_kwargs):
        """
        predict_kwargs can include:
            n_tiles, prob_thresh, nms_thresh, etc. 
            see StarDistBase.predict_instances
        
        """
        from copy import deepcopy
        
        predict_kwargs = deepcopy(predict_kwargs) # copy so popping doesn't alter attribute #TODO think can move to _predict method
        prediction_padding = None

        if 'prediction_padding' in predict_kwargs:
            prediction_padding = predict_kwargs.pop('prediction_padding')
            if prediction_padding is not None:
                assert isinstance(prediction_padding, int)
                input_array = np.pad(input_array, prediction_padding)

        if 'n_tiles' not in predict_kwargs or predict_kwargs.get('n_tiles') is None:
            predict_kwargs['n_tiles'] = self.model._guess_n_tiles(input_array)

        
        pred, info = self.model.predict_instances(input_array, **predict_kwargs)

        if prediction_padding:
            from SynAPSeg.utils.utils_image_processing import unpad
            pred = unpad(pred, prediction_padding)
        
        # free gpu memory
        del(info)
        from tensorflow.keras import backend as K
        K.clear_session()
        gc.collect()
        
        return pred

    def postprocess(self, pred, **postprocessing_kwargs):
        return pred.astype('int32') if (pred.max() <= (np.iinfo(np.int32).max)) else pred 
    
    @staticmethod
    def get_config(model_path):
        """ return stardist model config.json dict """
        import json
        cfg_path = os.path.join(model_path, 'config.json')

        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"config.json not found at: {cfg_path}")

        with open(cfg_path, 'r') as f:
            cfg = json.load(f)
        return cfg
    
    @staticmethod
    def get_model_indim_from_path(model_path):
        cfg = StardistModel.get_config(model_path)
        return cfg['ndim']
