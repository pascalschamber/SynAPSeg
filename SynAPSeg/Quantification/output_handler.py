from SynAPSeg.IO.BaseConfig import write_config
import os
import pandas as pd
from pandas.errors import EmptyDataError
import sys
from pathlib import Path
import shutil
import logging
from typing import Optional
import threading
import queue
import time

from SynAPSeg.utils import utils_general as ug
from SynAPSeg.IO.metadata_handler import MetadataParser
from SynAPSeg.utils.utils_ImgDB import ImgDB
from SynAPSeg.common.Logging import setup_default_logger

class OutputHandler:
    def __init__(self, stages, QUANT_CONFIG, logger=None, outdir_path=None):
        self.attach_logger(logger)
        self.main_container = {'all_summaries': []}
        self.data_keys = {'summary_df': 'all_summaries'} # points data keys to respective containers 
        self.init_containers(stages)
        self.QUANT_CONFIG = QUANT_CONFIG # attach quant config for reference
        self.cache_intermediate_results = self.QUANT_CONFIG['CACHE_INTERMEDIATE_RESULTS']
        
        # setup file structure for writing outputs
        self.set_outdir_path(outdir_path)
        
    def init_containers(self, stages):
        """
        expects:
        [{
            'container_name': 'all_soma_rpdfs',
            'container': [],
            'data_key': 'soma_rpdf'
        }]
        """
        for stage in stages:
            if not hasattr(stage, 'init_outputs'):
                continue

            output_configs = stage.init_outputs()
            for oc in output_configs:
                self.main_container[oc['container_name']] = oc['container']
                self.data_keys[oc['data_key']] = oc['container_name']
                
                   
                   
    def place_outputs(self, data, config):
        for key, cont_name in self.data_keys.items():
            result = data.get(key, pd.DataFrame())
            if self.cache_intermediate_results:
                self.main_container[cont_name].append(self.cache_results(key, cont_name, result, config))
            else:
                self.main_container[cont_name].append(result)
        self.logger.info(f'place_outputs completed.')
    

    def handle_writing_results_exceptions(self, key, df):
        if isinstance(df, pd.DataFrame):
            if key == 'rpdf' and 'coords' in df.columns:
                df['coords'] = df['coords'].apply(lambda x: x.tolist()) # trying to avoid potential error with saving np arrays, when they ahve to be converted to strings?
        return df
    
    def cache_results(self, key, cont_name, result, config):
        """ 
        save intermediate results to tempfolder, then load all and compile at end 
        
        returns: path to temp result
        """
        assert self.tempfolder is not None
        ex_id = Path(config.path_to_example).stem
        cache_dir = ug.verify_outputdir(os.path.join(self.tempfolder, cont_name))
        outpath = os.path.join(cache_dir, f"{cont_name}_{key}_{ex_id}.csv")
        result = self.handle_writing_results_exceptions(key, result)
        result.to_csv(outpath, index=False)
        self.logger.debug(f'cache_results complete for container:{cont_name} key:{key}.')
        return outpath
        
    def load_cached_results(self):
        """ 
        if self.cache_intermediate_results,
            read paths from main_container and update with lists of dataframes
            iters over each ex's data path in sub container 
            try to load cached data
                handles case where pipeline stage is attached but req. data keys are missing. 
                    e.g. roi_handling in pipeline.stages but conifg.FILE_MAP['ROIS'] is empty
                this would have written an empty df to disk, but reading it would throw an error so we just append an empty one instead
        updates 
            self.main_container with list[pd.DataFrame]
        """
        if not self.cache_intermediate_results:
            return
        
        from copy import deepcopy
        
        self.logger.info(f'loading cached results..')
        self.main_container_copy = deepcopy(self.main_container)
        _temp_container = {cont_name:[] for cont_name in self.main_container.keys()}

        for cont_name in self.main_container.keys():
            cache_dir = os.path.join(self.tempfolder, cont_name)
            cache_contents = ug.get_contents(cache_dir)

            for p in cache_contents:
                try:
                    df = pd.read_csv(p)
                except EmptyDataError:
                    df = pd.DataFrame()
                _temp_container[cont_name].append(df) 
            
        self.main_container = _temp_container

        self.logger.debug("\n\t".join(
            [f'loaded cached results.'] + \
                [f"{k}: n={len(v)}" for k,v in self.main_container.items()]))
        

    def concat_outputs(self, config):
        
        self.logger.info(f'concat_outputs started..')
        
        self.load_cached_results()
        
        self.main_container =  {
            k: pd.concat(v, ignore_index=False) for k,v in self.main_container.items()
        }
        self.logger.info(f'merged outputs.')
        
        if config.WRITE_OUTPUT:
            for fn, df in self.main_container.items():
                outpath = os.path.join(self.outdir_path, f"{fn}.csv")
                df.to_csv(outpath, index=False)
            self.logger.info(f'WRITE_OUTPUT completed.')
    
    def log_quant_config(self, proj, dispatchers, QUANT_CONFIG):
        try:
            import yaml
            from copy import deepcopy
            conf_out_fn = f"{ug.get_datetime()}_QUANT_CONFIG.yaml"
            conf_log_path = os.path.join(proj.configdir, conf_out_fn)
            
            # not that useful atm, probs want data['metadata']
            displogs = {}
            for dptcher in dispatchers:
                dispconf = ug.objdir(dptcher, return_as_dict=True)
                dispconf_unique_vals = {}
                for k,v in dispconf.items():
                    if k.startswith('__') or k=='config':
                        continue
                    if k in QUANT_CONFIG.params and v == QUANT_CONFIG.params[k]:
                        continue
                    if isinstance(v, type({}.keys())):
                        v = list(v) 
                    dispconf_unique_vals[k] = v
                displogs[dptcher.disp_i] = dispconf_unique_vals
            
            outparams = {
                '__desc__': 'contains quantification config as well as configs used by each dispatcher',
                'quant_config': deepcopy(QUANT_CONFIG.params),
                'dispatcher_configs': displogs,
            }
            
            with open(conf_log_path, 'w') as f: # save to projects log dir 
                yaml.dump(outparams, f, default_flow_style=False)

            with open(os.path.join(self.outdir_path, conf_out_fn), 'w') as f: # save to outputdir
                yaml.dump(outparams, f, default_flow_style=False)

        except Exception as e:
            print(f'unable to log config\nerror: {e}')

    def log_colocal_ids(self, data):
        
        image_channels, clc_nuc_info = MetadataParser.get_imgdb_colocal_nuclei_info(data['metadata'])
        imgdb = ImgDB(image_channels=image_channels, colocal_nuclei_info=clc_nuc_info)
        clc_id_outpath = os.path.join(self.outdir_path, 'colocal_ids.yaml')        
        write_config(imgdb.__dict__, clc_id_outpath)

    def print_main_container_head(self):
        for k,v in self.main_container.items():
            self.logger.info(f"\ncontainer key:{k}\n{'#'*40}\n{v.head(5)}\n")

    def attach_logger(self, logger):
        """ Logger setup using a shared logger or creates one if logger=None"""
        self._shared_logger = logger
        
        if self._shared_logger is not None:
            # Wrap the shared logger with a LoggerAdapter that injects stage="..."
            self.logger = logging.LoggerAdapter(
                self._shared_logger, {"stage": self.__class__.__name__}
            )
        else:
            self.logger = setup_default_logger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

    def set_outdir_path(self, outdir_path: Optional[str]=None):
        """ set directory for quant outputs, if outdir path is None auto generates a folder """
        if outdir_path is None:
            outdir_path = ug.verify_outputdir(
                os.path.join(
                    self.QUANT_CONFIG.OUTPUT_DIR_BASE, 
                    'outputs', 
                    f"{ug.get_datetime()}_Quantification_{self.QUANT_CONFIG.EXAMPLE_PROJ}")
                , makedirs=True)
            
        self.outdir_path = outdir_path
        self.QUANT_CONFIG.params['output_dir'] = outdir_path # update config with this info
        self.logger.info(f'set outdir_path:\n\t{outdir_path}')

        # setup folder for intermediate results
        self.tempfolder = None
        if self.cache_intermediate_results:
            self.tempfolder = ug.verify_outputdir(os.path.join(self.outdir_path, 'temp'), makedirs=True)

    def delete_tempfolder(self):
        """
        Delete the temporary folder used to cache intermediate results and all of its contents.
            note: only delete if all dispatchers have been run else concat outputs will fail
        """
        temp = self.tempfolder

        if isinstance(temp, str) and os.path.exists(temp):
            try:
                shutil.rmtree(temp)
                self.logger.info(f"Deleted temp folder contents: {temp}")
            except Exception as e:
                self.logger.error(f"Failed to delete temp folder {temp}: {e}")
                raise
        else:
            self.logger.warning(f"No temp folder to delete or invalid path: {temp}")



def prompt_delete_tempfolder(output_handler, timeout_duration: int = 8):
    """Prompt to confirm deleting tempdir, with a real timeout.
        TODO doesn't work inside jupyter, not connected to real delete function
    """
    q = queue.Queue()
    t0 = time.time()

    def get_input():
        try:
            user_in = input(
                "type 'yes' to confirm deletion of cache intermediate results temp dir\n"
                "to cancel type 'n': "
            )
            q.put(user_in)
        except Exception as e:
            q.put(e)

    # Start input in background so main thread can enforce timeout
    thread = threading.Thread(target=get_input, daemon=True)
    thread.start()

    while True:
        elapsed = time.time() - t0
        if elapsed >= timeout_duration:
            print(f"\nTimed out after {timeout_duration} seconds — canceled temp dir deletion.")
            return False  # or whatever you want on timeout

        try:
            # Poll for input every 0.5 s so we can update elapsed time
            result = q.get(timeout=0.5)
        except queue.Empty:
            # No input yet, keep waiting
            print(f"elapsed time: {elapsed:.1f} s", end="\r")
            continue

        # Handle errors from input thread
        if isinstance(result, Exception):
            print(f"\nError while reading input: {result}")
            return False

        confirm_delete_temp = result.strip().lower()
        if confirm_delete_temp == 'yes':
            print(f"\ndeleting temp dir: {output_handler.tempfolder}")
            return True
        elif confirm_delete_temp == 'n':
            print("\ncanceled temp dir deletion")
            return False
        else:
            print("\nInvalid input — please type 'yes' or 'n'.")
            # restart input thread for a new attempt
            q = queue.Queue()
            t0 = time.time()  # or keep original timeout; your call
            thread = threading.Thread(target=get_input, daemon=True)
            thread.start()
