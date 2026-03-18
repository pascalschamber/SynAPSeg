import os
import numpy as np

from SynAPSeg.models.factory import ModelPluginFactory

def get_model_pred(arr:np.ndarray, FMT:str = 'ZYX', pred_kwargs:dict=None, kwargs:dict=None):
    modelbasedir = os.environ['MODELS_BASE_DIR']
    default_models = {
        '3d':'StarDist_custom_3D', 
        '2d':'StarDist_custom_2D',
    }
    model_path = os.path.join(modelbasedir, default_models['3d' if len(FMT) == 3 else '2d'])

    par = {
        'model_path': model_path,
        "in_dims_model": FMT,
        "in_dims_pipe": FMT,
        "out_dims_pipe": FMT,
        'load_model_kwargs': {'weights_filename':'weights_last.h5'},
        "preprocessing_kwargs": {"norm": [1, (99.99 if len(FMT) == 3 else 99.9)]},
        'predict_kwargs':pred_kwargs or {},
    }
    par.update(kwargs or {})

    model = ModelPluginFactory.get_plugin('Stardist', **par)
    return model(arr)[0]