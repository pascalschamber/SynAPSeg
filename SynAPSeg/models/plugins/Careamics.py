import os
import sys
from pathlib import Path
import yaml
import numpy as np
import gc

from SynAPSeg.models.base import SegmentationModel
from SynAPSeg.utils import utils_general as ug


__plugin_group__ = "model"
__plugin__ = "CareamicsModel"
__parameters__ = "Careamics.yaml"

class CareamicsModel(SegmentationModel):

    def post_init(self):
        """ check n2v models exists, if not initiate training mode """
        
        model_path = self.model_path # should be dir to folder that contains models for each channel

        # for now just training on each image so setting self.model to skip loading step since model doesn't exist yet
        # TODO in the future can put logic here to check if input path exists and only do this if it doesn't
        # see prevous n2v lib implementation (n2v.py)

        self.model = 'placeholder'

    def load_model(self, model_path, **load_model_kwargs):
        pass 

    def _preprocess(self, img):
        """ CareamicsModel overrites base class method to preserve input image for training """
        kwargs = self.preprocessing_kwargs
        return self.preprocess(img, **kwargs)
    
    def preprocess(self, input_array, **preprocessing_kwargs):
        """
        Trains a model if post_init set's placeholder model
            Sneaking in training on the fly here, still returns the normalized image
            Normalizes the input to the range (0,1), using default args: norm=(1, 99.9), clip=True.
        """
        import csbdeep
        norm = preprocessing_kwargs.get('norm', (1, 99.9))
        clip = preprocessing_kwargs.get('clip', True)
        ch_axis = self.in_dims_pipe.index('C') if 'C' in self.in_dims_pipe else None # TODO: this should actually be like [1,2,3] if want to normalize along C axis of CZYX array
        train_mode = self.train_kwargs.get('train_mode') or 'single'
        
        # normalize
        arr = csbdeep.utils.normalize(input_array, pmin=norm[0], pmax=norm[1], axis=[3,4,5], clip=clip)

        # maybe train on input 
        if train_mode=='single':
            import torch
            if hasattr(self, 'model'):
                del(self.model)
            if hasattr(self, 'n2v_config'):
                del(self.n2v_config)
            torch.cuda.empty_cache()
            gc.collect()
            self.model = self.train(arr, self.train_kwargs)

        return arr 
    
    def train(self, image, conf):
        """ train a model """
        from careamics.config import create_n2v_configuration  # Noise2Void
        from careamics import CAREamist
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        from SynAPSeg.utils import utils_image_processing as uip
        
        # to add multi-image training support should pass arg `data` as list of STCZYX arrays
        # which is already cropped
        # TODO: full support for ST dims is not currently implemented, if present will just take first index on these axes

        # default params (non caremics config) 
        crop_size = conf.get('crop_size')                   # interp crop_size == None to be full image after getting input dim size 
        # crop_size = [512, 512]  
        # [16, 1024, 1024]

        take_dims = uip.subtract_dimstr(self.in_dims_pipe, self.in_dims_model)
        project_dims = ''
        n_channels = 1 if 'C' not in self.in_dims_pipe else image.shape[self.in_dims_pipe.index('C')]
        plot_train_history = conf.get('plot_train_history', True) 

        # parse relevant state params
        data_state = self.data_state or {}
        ex_dir = data_state.get('path_to_example')
        if ex_dir is not None: 
            ug.verify_outputdir(ex_dir)
        

        # ingest input - should already be normalized in this case #TODO this won't always be true
        train_data, imgfmt = uip.reduce_dimensions(
            image, 
            self.in_dims_pipe, 
            take_dims=take_dims, 
            project_dims=project_dims, 
            return_current_format=True
        )

        imgdimsizes = dict(zip(imgfmt, train_data.shape))
        ch_axis = imgfmt.index('C') if 'C' in imgfmt else None
        spaital_dims = imgfmt.replace('C', '')
        print(f'post-reduce_dimensions image.shape: {image.shape}, imgdimsizes:{imgdimsizes}')
        
        
        # generate training crop 
        #########################
        crop_size = crop_size or [imgfmt.index(d) for d in spaital_dims]     # if None, trains on whole image 
        # handle input spec validation (e.g. crop dims align with image dim)
        if len(crop_size) != len(spaital_dims):
            raise ValueError(f"crop size ({crop_size}) must have same number of dims as spaital_dims ({spaital_dims})")
        crop_dimsizes = dict(zip(spaital_dims, crop_size)) # !!!

        rand_nd_slice = [] 
        for dim, s in imgdimsizes.items():
            if dim == 'C':                  # take all present channels
                slc = slice(0,s)
            else: # get a random start point within image bounds to crop 
                st = np.random.randint(0, s - crop_dimsizes[dim])
                slc = slice(st, st+crop_dimsizes[dim])
            rand_nd_slice.append(slc)
        tc = train_data[tuple(rand_nd_slice)]
        print('post generate training crop, train data array info:')
        uip.pai(tc)

        # update default careamics params with users self.train_kwargs (conf)
        conf_params = self._get_default_train_params(conf, n_channels=n_channels)
        print(f'in train image.shape: {image.shape}\nusing conf_params:\n{conf_params}')

        self.n2v_config = create_n2v_configuration(**conf_params)
        print(f"self.n2v_config:\n{self.n2v_config}")

        # instantiate a careamist
        careamist = CAREamist(self.n2v_config, work_dir=ex_dir)
        
        # train
        careamist.train(train_source=tc)

        if plot_train_history:
            histdf = pd.DataFrame(data=careamist.get_losses())
            sns.lineplot(histdf, x='train_epoch', y='train_loss')
            sns.lineplot(histdf, x='val_epoch', y='val_loss')
            plt.show()
        
        # save the model config to the example
        if ex_dir is not None:
            with open(os.path.join(ex_dir, f"{self.name}_config.yaml"), 'w') as f:
                yaml.dump(self.n2v_config.model_dump(), f)

        return careamist


    def _get_default_train_params(self, train_params, n_channels=1):
        """ note need to update n_channels based on train input shape """
        axes = self.in_dims_model
        n_channels = n_channels or 1 # think must always be 1

        return dict(
            experiment_name="n2v2",
            data_type="array",
            axes=axes,                  # Axes of the data (e.g. SYX).
            n_channels=n_channels,
            
            patch_size=                 train_params.get('patch_size') or [16, 64, 64][-(len(axes) -1):],  # -1 for C axis
            batch_size=                 train_params.get('batch_size'), 
            num_epochs=                 train_params.get('epochs'),
            use_n2v2=                   train_params.get('use_n2v2'),
            roi_size=                   train_params.get('roi_size'),  
            masked_pixel_percentage =   train_params.get('masked_pixel_percentage'),

            # these should be made into params 
            # independent_channels
            
            # given issues, these are best left as lib's defaults which are identical to below
            # train_dataloader_params={'shuffle': True},
            # val_dataloader_params={}, 

            # seems to be issue with progressbar stalling when setting num_workers 
            # train_dataloader_params={
            #     "num_workers": 2,  
            # },
            # val_dataloader_params={
            #     "num_workers": 2,  
            #     "persistent_workers": True,
            # },
        )



    def predict(self, input_array, **predict_kwargs):
        from SynAPSeg.utils import utils_image_processing as uip
        
        kwargs = predict_kwargs
        tile_size =    kwargs.get('tile_size')    or self.n2v_config.data_config.patch_size
        tile_overlap = kwargs.get('tile_overlap') or list(np.array(self.n2v_config.data_config.patch_size)//4)
        batch_size =   kwargs.get('batch_size')   or self.n2v_config.data_config.batch_size
        
        # once trained, predict
        pred = self.model.predict( # type: ignore
            source=input_array,  
            axes=self.in_dims_model,
            tile_size=tile_size, 
            tile_overlap= tile_overlap,
            batch_size=batch_size,
        )
        curr_fmt = 'TCYX' if pred[0].ndim==4 else 'TCZYX' # reduce dims since model returns leading extra dim
        pred, fmt = uip.collapse_singleton_dims(pred[0], curr_fmt) 
        assert fmt == self.in_dims_model, f"pred result format: {fmt} != self.in_dims_model: {self.in_dims_model}"

        import torch
        torch.cuda.empty_cache()
        gc.collect()

        return pred
    

