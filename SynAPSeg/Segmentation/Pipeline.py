import os
import sys
import traceback
import numpy as np

from SynAPSeg.utils import utils_image_processing as uip
from SynAPSeg.utils import utils_plotting as up

from SynAPSeg.models.factory import ModelPluginFactory
from SynAPSeg.models.base import SegmentationModel


def build_segmentation_pipeline(config):
    """ constructor function for building a pipeline from a config object """
    model_params = config['MODEL_PARAMS']
    models = []
    for model_name, params in model_params.items():
        model_type = params.get('plugin_class') or model_name
        try:
            models.append(ModelPluginFactory.get_plugin(model_type, **params))
        except Exception as e:
            tb = traceback.format_exc()
            raise ValueError (f"Error initializing model ({model_type}) with params:\n\t{params}\n{tb}")
    pipeline = SegmentationPipeline(models)
    return pipeline


class SegmentationPipeline:
    """
    A pipeline that orchestrates multiple segmentation models
    and handles dimension transformations as needed.
    """
    def __init__(self, models: list):
        self.models = []        # List of tuples: (model_name, model_instance)
        self.model_names = set() # Set of str 
        self.connections = {}   # Dictionary: source_model_name -> [target_model_names]
        self.data_state ={} #  holds meta data for the current run, gets passed to models
        
        # add models and connections
        for m in models:
            self.add_model(m)
    
    
    def __str__(self):
        astr = f"{type(self)}\n" + '`' * len(str(type(self))) + '\n'
        for k,v in self.__dict__.items():
            astr += f"\t{k}: {v}\n"
        return astr
    
    def __repr__(self):
        return self.__str__()
    
            
    def get(self, attr, return_if_not_found=None):
        if hasattr(self, attr):
            return getattr(self, attr)
        return return_if_not_found
    
    def get_model_names(self):
        """ get the names of all models that have been added """
        return self.model_names
    
    def get_model(self, model_name: str):
        """ get a model by name"""
        for name, model in self.models:
            if name == model_name:
                return model
        raise ValueError(f"cannot find {model_name} in models ({self.get_model_names()})")
            
        
    def add_model(self, model: SegmentationModel):
        """
        Add a segmentation model to the pipeline.

        Args:
            model (SegmentationModel): An instance of a segmentation model.
        """
        name = model.get('name')
        assert name is not None, "must pass a name in params during model init"
        assert name not in self.model_names, f"name must be unique. got {name}, already have {self.model_names}"
        self.models.append((model.name, model))
        self.model_names.add(name)
        
        # add any connections
        input_model_name = model.get('model_input')
        if input_model_name is not None:
            self.add_connections(input_model_name, name) 

    def add_connections(self, source_model: str, target_model: str):
        """
        Add a connection between models.

        Args:
            source_model (str): Name of the source model whose output will be passed.
            target_models (list): List of target model names that will receive the output.
        """
        if source_model not in self.connections:
            self.connections[source_model] = []
        self.connections[source_model].append(target_model)
    
    def validate_connections(self):
        """ensure models exist in self.connections if they provide input to another model, and that the targets exist"""
        for input_model, target_models in self.connections.items():
            assert input_model in self.model_names, f"undefined model connection:\n{input_model} not in models({self.model_names}), but is set as input for a model (connections: {self.connections})"
            for name in target_models:
                assert name in self.model_names, f"{name} not in self.model_names ({self.model_names})"

        for mn, m in self.models:
            m.update_state(self.data_state)


    def run(self, data: list, return_model_inputs=False, data_state={}, progress_callback=None) -> dict[str, list[np.ndarray]]:
        """
        Run the entire pipeline on the input data.

        Args:
            data (list): Input data to the pipeline.
            return_model_inputs (bool): control whether inputs to each model ('intermediate_data') are returned with results
            data_state (dict): optional, pass metadata about current run to models, passed with validate_connections
            progress_callback (callable): optional, callback function to report progress

        Returns:
            dict: Outputs from all the final models in the pipeline. keys are model names, values are model outputs
        """
        self.data_state = data_state

        # verify connections
        self.validate_connections()
            
        # Store the outputs of each model
        outputs = {}
        
        # Keep track of the intermediate data to feed into each model
        intermediate_data = {model_name: None for model_name, _ in self.models}

        # Initialize the first model(s) with the raw input data (i.e., models that are not listed as targets)
        all_targets = set()
        for targets in self.connections.values():
            all_targets.update(targets)

        for model_name, model in self.models:
            if model_name not in all_targets:
                intermediate_data[model_name] = data

        # Process each model in the defined order
        for i, (model_name, model) in enumerate(self.models):
            
            if progress_callback: # Report internal progress, caps at 99% so that 100% is reserved for writing outputs
                progress_callback(min(int((i / len(self.models)) * 100), 99), f"Running {model_name}...")


            if intermediate_data[model_name] is not None:
                print(f"applying {model_name}")
                # We assume the data is currently in the *output* dims
                # of whichever model produced it. If it's the first model,
                # we assume the user-provided data is already in the correct format.
                
                # For demonstration, we simply assume no transformation is needed
                # for the first model. For subsequent models, we transform from the
                # previous model's output dims to the current model's input dims.
                # That logic can be improved by storing the actual previous model's
                # output dims. Here we make a simplifying assumption.

                transformed_input_data = intermediate_data[model_name]

                # 2. Run the model
                outputs[model_name] = model(transformed_input_data)

                # 3. Transform the model's *output* to match each target model's input dims
                model_output_dims = model.get_output_dims()
                
                # Pass data along to connected models
                if model_name in self.connections:
                    for target_model_name in self.connections[model_name]:
                        # We retrieve the target model to get its input dims
                        target_model = next(
                            m for mn, m in self.models if mn == target_model_name
                        )
                        target_input_dims = target_model.get_input_dims()

                        # Transform each output to the target model's input dims
                        transformed_output_data = [
                            uip.transform_axes(
                                arr,
                                current_format=model_output_dims,
                                target_format=target_input_dims
                            )
                            for arr in outputs[model_name]
                        ]

                        # If the target model already has some data waiting, combine them
                        if intermediate_data[target_model_name] is None:
                            intermediate_data[target_model_name] = transformed_output_data
                        else:
                            intermediate_data[target_model_name] += transformed_output_data
        
        if progress_callback:
            progress_callback(100, "Pipeline run complete")
            
        if return_model_inputs:
            return outputs, {k:v for k,v in intermediate_data.items() if v is not None}
        return outputs



if __name__ == '__main__':
    
    # testing pipeline 
    ############################################################################################
           

    # loading real models without a config file
    if bool(0):
        
        n2v_model = ModelPluginFactory.get_plugin(
            'N2V', 
            multi_image = True,
            model_path = r'D:\BygraveLab\ConfocalImages\SEGMENTATION_DATASETS\2025_0103_1st_cohort_gpPV_63x\examples\n2v',
            in_dims_model = "ZYX",
            out_dims_pipe = "STCZYX", 
            name = 'n2v',
            load_model_kwargs = {'model_base_name':'N2V_3d'},
            preprocessing_kwargs = {'axes': 'STC', 'dtype': 'float32', 'norm': (1, 99.8)},
            predict_kwargs = {'n_tiles':(1,8,8,1)},
        )

        stardist_model = ModelPluginFactory.get_plugin(
            'stardist',
            model_path = r"D:\OneDrive - Tufts\Classes\Rotation\BygraveLab\BygraveCode\models\synapsedist_v3.4.3_noAug",
            in_dims_model="YX", 
            out_dims_pipe="STCYX", 
            name = "stardist",
            model_input = 'n2v',
            preprocessing_kwargs = {'axes': 'STC', 'dtype': 'float64', 'norm': (1, 99.8)},
            predict_kwargs = {'scale':1.0},
        )

        neurseg_model = ModelPluginFactory.get_plugin(
            'neurseg',
            model_path = r"D:\OneDrive - Tufts\Classes\Rotation\BygraveLab\BygraveCode\models\dentrite_models\2024_1206_181224_compiled__densenet201-valIOU0.5459-183.keras",
            in_dims_model="YX", 
            out_dims_pipe="STCYX",
            name = "neurseg",
            model_input = 'n2v',
            # dimension_handling = {'Z': "utils_image_processing.mip", 'C': "utils_image_processing.to_grayscale"}, # e.g. if wanted to convert n-ch image to grayscale
            load_model_kwargs = {'backbone': 'densenet201'},
            preprocessing_kwargs = {'axes': 'STC', 'dtype': 'float64', 'norm': (1, 99.8)},
            predict_kwargs = {'patch_shape': (256, 256), 'n_classes': 1},
            postprocessing_kwargs = {'remove_small_objs_size': 1000},
        )
        
        pipeline = SegmentationPipeline([n2v_model, stardist_model, neurseg_model])



        import tifffile
        img_p = r"D:\BygraveLab\ConfocalImages\SEGMENTATION_DATASETS\2025_0103_1st_cohort_gpPV_63x\examples\0001\raw_img.tiff"
        img = tifffile.imread(img_p)
        uip.pai(img)
        print(img.nbytes)

        img = uip.transform_axes(img, "CZYX", "STCZYX")
        uip.pai(img)
        print(img.nbytes)
        
        # Run the pipeline
        results = pipeline.run([img])
        
        for k,v in results.items():
            print(k)
            print(uip.pai(v))
            up.show_ch(v[0][0,0], axis=0)