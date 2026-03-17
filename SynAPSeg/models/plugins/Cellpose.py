from SynAPSeg.models.base import SegmentationModel

__plugin_group__ = "model"
__plugin__ = "CellposeModel"
__parameters__ = 'Cellpose.yaml'
__append_docs__ = {
    'predict': 'cellpose.models.CellposeModel.eval',
    'preprocess': 'csbdeep.utils.normalize'
}

class CellposeModel(SegmentationModel):
    def post_init(self):
        self._append_docs(__append_docs__)

    def load_model(self, model_path, **load_model_kwargs):
        from cellpose.models import CellposeModel as cpmodel
        
        gpu = load_model_kwargs.get('gpu', True)
        assert len(self.in_dims_model) in [2,3], f"in_dims_model must be of length 2 or 3 but got: {self.in_dims_model}"
        
        return cpmodel(
            gpu=gpu, 
            pretrained_model=model_path
        )

    def preprocess(self, input_array, **preprocessing_kwargs):
        """
        Normalizes the input to the range (0,1).
        """
        from SynAPSeg.utils.utils_image_processing import norm_percentile

        norm = preprocessing_kwargs.get('norm', (1, 99.9))
        clip = preprocessing_kwargs.get('clip', False)
                
        return norm_percentile(input_array, norm=norm, ch_axis=None, clip=clip)
    
    def predict(self, input_array, **predict_kwargs):
        """
        Wrapper for 'CellposeModel.eval()'.
        """

        res = self.model.eval(input_array, **predict_kwargs or {}) # returns masks, [plot.dx_to_circ(dP), dP, cellprob], styles 
        return res[0] # <-- we just want the mask
        
        

