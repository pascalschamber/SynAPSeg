import os
import sys
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Optional
import importlib
from inspect import getdoc

from SynAPSeg.utils import utils_image_processing as uip
from SynAPSeg.utils import utils_general as ug



class SegmentationModel(ABC):
    """
    Abstract base class for segmentation models.
    """

    def __init__(
        self, 
        model_path: str,
        in_dims_model: str, 
        out_dims_pipe: str, 
        in_dims_pipe: str = 'STCZYX', # TODO can infer when calling model.run if image has same n dims as in_dims_model this has same format
        out_dims_model: str = '',
        name: Optional[str] = None,
        model_input: Optional[str] = None,
        dimension_handling: Optional[dict] = None, 
        expand_result: bool = True,
        debug: bool = False,
        default_reduce_fxn: Optional[Callable] = None, 
        preprocessing_kwargs: Optional[dict] = None,
        load_model_kwargs: Optional[dict] = None,
        predict_kwargs: Optional[dict] = None,
        postprocessing_kwargs: Optional[dict] = None,
        train_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initialize the segmentation model with parameters and dimension formats.

        Args:
            in_dims_model (str): Expected input dimension format for model.predict (e.g., for a 2d model 'YX').
            out_dims_pipe (str): Output dimension format after iterating over non-reduced dimensions. 
                Out_dims_pipes allows for control of how the input in fed into the model.
                Dimensions specified here, that are not part of in_dims_model, will be iterated over.
                Unspecified dimensions will be flattened by projection automatically.
                    Note: singleton dims are preseved to maintain constistent format.
                        
                Example: 
                    Given a 3d volume as input (in_dims_pipe='ZYX'), applying a 2d model (in_dims_model=YX), 
                    which generates a 2d output (out_dims_model=YX)..
                    To run the prediction on the z-projection of the image, 
                        out_dims_pipe = 'YX'
                    While if you wanted to apply the same model slice-by-slice over the volume,
                        out_dims_pipe = 'ZYX'
            
            name (str): must be passed here if using in a segmentation pipeline
            model_input (str): if using as a pipeline, and is not None or assumes input is derived from another model
            dimension_handling (dict(dim_str: callable | str), or None): define functions to apply when reducing specific dimensions. if a string tries to load function using importlib.
                Otherwise, the default_reduce_fxn is applied (maximum intensity projection).
                The passed function must have a kwarg for axis=x, which gets passed from the index of the dim_str.
            kwargs (dict): Dict containing parameters e.g. for preprocessing, prediction, postprocessing
        """
        self.in_dims_model = in_dims_model
        self.out_dims_pipe = out_dims_pipe
        self.name = name
        self.model_input = model_input
        self.model_path = model_path if model_path is not None else kwargs.get('output_dir')
        self.model = None # dynamically loaded during prediction
        self.img_i = None # store index of current img
        self.data_state ={} #  holds meta data for the current run so models can be context aware
        
        # options
        self.dimension_handling = dimension_handling or {}
        self.expand_result = expand_result # controls whether dimensions of result returned by self.run should be filled with singleton dimensions that were reduced during prediction
        self.debug = debug
        
        # will be inferred if not directly specified, # for all except in_dims_pipe, out_dims_model. not sure they even need to be optionally set as they should be set based on these params anyways
        self.in_dims_pipe = in_dims_pipe
        self.out_dims_model = out_dims_model or ''
        self.iter_array_dims = ''
        self.iter_dims = ''
        self.reduce_dims = ''
        
        # potentially set by individual models
        self.default_reduce_fxn = default_reduce_fxn or uip.mip        
        self.preprocessing_kwargs = preprocessing_kwargs or {}
        self.load_model_kwargs = load_model_kwargs or {}
        self.predict_kwargs = predict_kwargs or {}
        self.postprocessing_kwargs = postprocessing_kwargs or {}
        self.train_kwargs = train_kwargs or {}
        
        for k,v in kwargs.items():
            setattr(self, k, v)
        
        self._docs_appended = False
        self.post_init()
        
        
    def post_init(self):
        """ can override if need custom behavior here """
        pass                
        
    def __call__(self, data: list):
        """ Allow the model to be called like a function """
        return self.run(data)
    
    def get(self, attr, return_if_not_found=None):
        if hasattr(self, attr) and (getattr(self, attr) is not None):
            return getattr(self, attr)
        return return_if_not_found
    
    def get_input_dims(self):
        """ 
        Returns the dimension format of the pipeline input. 
        """
        return self.in_dims_pipe
    
    def get_output_dims(self):
        """ 
        Returns the dimension format of the pipeline output. 
        If expand_result is True, in_dims_pipe is returned. Otherwise, out_dims_pipe is returned.
        """
        if self.expand_result:
            return self.in_dims_pipe
        return self.out_dims_pipe

    def update_state(self, data_state):
        """ 
        Allows data context to be passed in during pipeline run. 
        """
        self.data_state = data_state
    
    
    def apply_dimensionality_reductions(self, arr: np.ndarray, current_format: str):
        """ 
        Apply all reductions, using function declared in dimention_handling, otherwise default_reduce_fxn
        """
        # apply reductions in order specified by dimension_handling, when provided 
        self.reduce_dims_order = "".join([d for d in self.dimension_handling.keys()])
        # add any dims that were not explicitly specified if dimension_handling
        for d in self.reduce_dims:
            if d not in self.reduce_dims_order:
                self.reduce_dims_order += d
        # iter the dims, apply reduction and updating the current format
        for d in self.reduce_dims_order:
            assert d in self.reduce_dims, f"ensure {d} is intentionally defined in dimension handling"
            if self.debug: print(f"in apply_dimensionality_reductions. applying reduction on dim: {d}, current_format:{current_format}, shape: {arr.shape}")
            arr = self.reduce_dim(arr, current_format, d, reduce_fxn = self.dimension_handling.get(d, None))
            current_format = current_format.replace(d, '')
            if self.debug: print(f"in apply_dimensionality_reductions. result current_format:{current_format}, shape: {arr.shape}")
        return arr, current_format

    def reduce_dim(self, arr: np.ndarray, current_format: str, dim: str, reduce_fxn=None):
        """ 
        Apply dimensionality reduction on a single axis
        arr: np.array
        current_format: dimenstion string (e.g. 'ZYX')
        dim: len(1) dimenstion string to reduce (e.g. 'Z'), 
        reduce_fxn: function to apply (must accept kwarg axis=int)
        """
        assert isinstance(dim, str) and len(dim) == 1
        reduce_fxn = self.default_reduce_fxn if reduce_fxn is None else ug.get_function(reduce_fxn) if not callable(reduce_fxn) else reduce_fxn
        return reduce_fxn(arr, axis=current_format.index(dim))
            
    
    def handle_axes(self, img: np.ndarray, current_format=None):
        """ 
        Determines how each dimension in the input pipeline (in_dims_pipe) is handled during prediction. 
        Dimensions are categorized as part of the model's input/output, reduced, or iterated over, 
        guiding how the input image is processed and predictions are mapped to the output.
        
        Args:
            img (numpy.ndarray): The input image array to be processed. 
                Must have dimensions matching the length of `self.in_dims_pipe`.
        
        Notes:
            - If `iter_array_dims`, `iter_dims`, `reduce_dims`, or `out_dims_model` 
              are not explicitly set during initialization, they are derived here.
        
        Example:
            If `in_dims_pipe` is "CZYX", `in_dims_model` is "YX", and `out_dims_pipe` 
            is "CYX", the method determines which dimensions are iterated over (e.g., C),
            reduced (e.g., Z), and how the output shape maps back to the pipeline's output format.
        """
        # check img shape matches expectation
        img_shape = img.shape
        current_format = self.in_dims_pipe if current_format is None else current_format
        assert img.ndim == len(current_format), f"input array shape ({img.shape}) doesn't match expected format ({current_format}), may need to change in_dims_pipe"

        
        # determine how axes will be handled
        axd = dict(    
            iter_array_dims = '', # format of array that will be iterated over
            iter_dims       = '', # dimensions that will be iterated
            reduce_dims     = '', # dimensions that will be reduced before iteration
            out_dims_model  = '', # infer what dims are consumed during model predict e.g. if takes ZYX but only returns YX
        )
        for d in current_format:
            if d in self.in_dims_model:
                if d in self.out_dims_pipe: 
                    axd['out_dims_model'] += d
                axd['iter_array_dims'] += d
            else:
                if d not in self.out_dims_pipe:
                    axd['reduce_dims'] += d
                else:
                    axd['iter_dims'] += d
                    axd['iter_array_dims'] += d
                    
        # if not explicitly set during __init__, set out_dims_model instead of using inferred version
        overrideable = ['out_dims_model'] # think this is only one, since need to reset 'iter_array_dims' after training
        for dim_attr, dim_val in axd.items():
            if dim_attr in overrideable:
                if len(getattr(self, dim_attr)) == 0: # only set if not provided in init
                    setattr(self, dim_attr, dim_val)
            else:
                setattr(self, dim_attr, dim_val)
        
        # axis indicies for iteration
        self.iter_axes = tuple([self.iter_array_dims.index(d) for d in self.iter_dims])       
        
        # Create an array to store the prediction results in the pipeline output shape
        self.out_shape = [img_shape[current_format.index(dim)] for dim in self.out_dims_pipe] # TODO doesn't handle diff model in/out dims in some cases

    
    
    def run(self, input_data):
        """ 
        Perform prediction on a list of n-dimensional arrays.
        
        Args:
            input_data (list): list of arrays.
        
        Returns:
            list: list of Processed image with the format defined by `out_dims`.
        """
        input_data = [input_data] if not isinstance(input_data, list) else input_data
        assert isinstance(input_data[0], np.ndarray), f"expected array but got {type(input_data[0])}"
        
        results = []
        for img_i, img in enumerate(input_data):
            if self.debug:
                from SynAPSeg.utils.utils_ML import cuda_mem_alloc
                m0 = cuda_mem_alloc()
                
            self.img_i = img_i
            self.handle_axes(img) # determine axes for predicting and generating output, running here so preproc function can use some of the attributes
            
            if self.model is None:
                self._load_model()
            
            _img = self._preprocess(img)       
            _img = self._predict(_img)
            _img = self._postprocess(_img)
            results.append(_img)

            if self.debug:
                print(f"{self.name} -- gpu mem pre/post run: {m0} --> {cuda_mem_alloc()}")
        return results
        
    def _load_model(self):
        model_path = self.model_path
        kwargs = self.load_model_kwargs or {}
        self.model = self.load_model(model_path, **kwargs)
        
    @abstractmethod
    def load_model(self, model_path: str, **load_model_kwargs):
        pass
    
    
    def _preprocess(self, img):
        """ wrap model specific preprocessing function and apply it over img """
        kwargs = self.preprocessing_kwargs
        default_iter_axes = "".join([d for d in self.out_dims_pipe if d not in self.in_dims_model]) # all dims not used as input to model
        preproc_axes = kwargs.get('axes', default_iter_axes)
        preproc_dtype = kwargs.get('dtype', 'float64')
                
        return uip.apply_to_chunks(
            img, self.in_dims_pipe, preproc_axes, 
            self.preprocess, out_dtype=preproc_dtype,
            **kwargs
            )
    
    def preprocess(self, input_array: np.ndarray, **preprocess_kwargs):
        """
        Default implementation of preprocess, returns the input. Subclasses should override this if they need custom behavior.
        """
        return input_array

        
    def _predict(self, img: np.ndarray):
        """
        Perform prediction on a single n-dimensional array.

        Args:
            img (np.ndarray): Input array of shape 'STCZYX'.
        
        Returns:
            np.ndarray: Processed array with the shape defined by `out_dims`.
        """                    
        # determine axes for predicting and generating output
        result_array = result_array = np.zeros(self.out_shape, dtype=img.dtype)
        if self.debug: print('result_array.shape:', result_array.shape)
        # track when first prediction is generated to set the result_array dtype based on it
        pred_generated = False
        
        inp_arr, self.inp_format = self.apply_dimensionality_reductions(img, self.in_dims_pipe)
        if self.debug: print('inp_arr', inp_arr.shape)
        
        # iterate over remaining axes that are not part of model input
        self.iter_shape = tuple(inp_arr.shape[self.iter_array_dims.index(d)] for d in self.iter_dims) # shape of iter_axes
        self.iter_indicies = np.ndindex(self.iter_shape)
        
        for indices in self.iter_indicies: 
            if self.debug: print(f"iter_axes: {self.iter_axes}, iter_dims: {self.iter_dims}, indices: {indices}")
            self.current_indicies = indices
            
            in_slice_index = uip.index_along_axes(inp_arr, self.iter_axes, indices, return_indexer=True)
            input_slice = inp_arr[in_slice_index]
            
            pred = self.predict(input_slice, **self.predict_kwargs)
            if pred_generated is False:
                result_array = result_array.astype(pred.dtype.name)
            
            out_slice_index = uip.index_along_axes(result_array, self.iter_axes, indices, return_indexer=True)
            result_array[out_slice_index] = pred
            pred_generated = True
        
        if self.expand_result:
            result_array = uip.transform_axes(result_array, current_format=self.out_dims_pipe, target_format=self.in_dims_pipe)
            
        return result_array
    
    @abstractmethod
    def predict(self, input_array: np.ndarray, **predict_kwargs) -> np.ndarray:
        pass

    
    def _postprocess(self, img):
        """ wrap model specific postprocessing function and apply it over img """
        kwargs = self.postprocessing_kwargs
        default_iter_axes = "".join([d for d in self.in_dims_pipe if d not in self.out_dims_model])
        postproc_axes = kwargs.get('axes') or default_iter_axes
        outdtype = kwargs.get('outdtype', 'int32')
        return uip.apply_to_chunks(img, self.in_dims_pipe, postproc_axes, self.postprocess, out_dtype=outdtype)
    
    def postprocess(self, input_array: np.ndarray, **postprocessing_kwargs):
        """
        Default implementation of postprocess to handle segmentation outputs
            removes small objects (relabeling if so)
        Subclasses should override this if they need custom behavior.
        """
        
        remove_small_objs_size = postprocessing_kwargs.get('remove_small_objs_size', None)
        connectivity = postprocessing_kwargs.get('connectivity', 1)
        preserve_labels = postprocessing_kwargs.get('preserve_labels', False) 
        # instance segmentation won't care if input is relabeled
        # others where label value is class type need to keep same labels (e.g. for neurseg)

        if remove_small_objs_size is not None:
            
            labeled_img = uip.relabel(input_array)
            cleaned_img = uip.remove_small_objs(labeled_img, min_size=remove_small_objs_size, connectivity=connectivity)
            if preserve_labels:
                return np.where(cleaned_img > 0, input_array, 0)
            else:
                return uip.relabel(cleaned_img)
            
        return input_array


    def __str__(self):
        astr = f"{type(self)}\n" + '`' * len(str(type(self))) + '\n'
        for k,v in self.__dict__.items():
            astr += f"\t{k}: {v}\n"
        return astr
    
    def __repr__(self):
        return self.__str__()

    @classmethod
    def _append_docs(cls, doc_map: dict[str, str]):
        """
        Dynamically appends external docstrings to child class methods.
            be sure to call in post_init of child class to preserve this base classes's docstrings
        
        Args:
            doc_map (dict[str, str]): mapping of {local_method_name: external_class_path}
        """
        # Ensure we only perform this expensive operation once per class type
        if getattr(cls, "_docs_appended_to", None) == cls.__name__:
            return

        for local_name, external_path in doc_map.items():
            try:
                # 1. Resolve the external object
                parts = external_path.split('.')
                target_obj = None
                
                for i in range(len(parts) - 1, 0, -1):
                    mod_name = ".".join(parts[:i])
                    attr_path = parts[i:]
                    try:
                        module = importlib.import_module(mod_name)
                        target_obj = module
                        for attr in attr_path:
                            target_obj = getattr(target_obj, attr)
                        break 
                    except (ImportError, AttributeError):
                        continue

                if target_obj:
                    # 2. Resolve the local method on the subclass
                    local_method = getattr(cls, local_name)
                    ext_doc = getdoc(target_obj)
                    
                    if ext_doc:
                        separator = f"\n\n{'-'*40}\nInherited Documentation from {external_path}:\n"
                        # Apply doc to the method of the subclass
                        local_method.__doc__ = (getdoc(local_method) or "") + separator + ext_doc
            
            except Exception as e:
                # Fail silently or log to prevent breaking the main pipeline
                print(f"DEBUG: Could not append docs for {cls.__name__}.{local_name}: {e}")

        # Mark this specific class as updated
        cls._docs_appended_to = cls.__name__

