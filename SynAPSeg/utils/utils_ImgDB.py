from copy import deepcopy

class ImgDB:
    """
    Class for managing image data and colocalization parameters.
        represents the each channel with a base colocal_id and a channel index.
        colocalizations are constructed from intersecting arbitrary number of base colocal_ids.

    Attributes:
        image_channels (list of dict): base colocal_ids which map directly to image channels
            [{'name': 'MAP2', 'ch_idx': 0, 'colocal_id': 0}, 
            {'name': 'Traf3-eGFP', 'ch_idx': 1, 'colocal_id': 1},
            {'name': 'Psd95', 'ch_idx': 2, 'colocal_id': 2},]
        
        colocalid_ch_map (dict[int, int]): map of base colocal_id to channel index
            example: {0: 0, 1: 1, 2: 2}

        colocalizations (list[dict]): colocalized ids constructed from intersecting base colocal_ids.
            example:
            [{'name': 'Traf3-eGFP+Psd95', 'coChs': [1, 2], 'coIds': [1, 2], 'assign_colocal_id': 3,
            'intersecting_label_columns': ['ch1_intersecting_label'], 'intersecting_colocal_ids': [1],
            'base_colocal_id': 2}]

        colocal_nuclei_info (list of dict): # DEPRECATED, replaced by colocalizations however has key names that align with image_channels
            [{'name': 'Traf3-eGFP+Psd95', 'ch_idx': [1, 2], 'co_ids': [1, 2], 'colocal_id': 3}, 
            {'name': 'MAP2+Traf3-eGFP+Psd95', 'ch_idx': [0, 1, 2], 'co_ids': [0, 1, 2], 'colocal_id': 4}]

    """

    def __init__(self, image_channels, colocal_nuclei_info , **kwargs):
        self.image_channels = image_channels
        self.colocal_nuclei_info = colocal_nuclei_info 
        self.colocalid_ch_map = {} # should rename to ch_colocalid_map
        self.colocal_ids = {}
        self.colocalizations = [] # 
        self.colocalized_ids = [] # colocal ids assigned to one of the colocalizations
        self.ingest_kwargs()
    
    def ingest_kwargs(self):
        # get assigned mapping of image channels to colocalids
        for ch_dict in self.image_channels:
            clc_id = self.check_not_none(ch_dict.get("colocal_id"))
            assert clc_id not in self.colocal_ids, f"colocal_id must be unique, got {clc_id} but have {self.colocal_ids}"
            self.colocalid_ch_map[self.check_not_none(ch_dict.get("ch_idx"))] = clc_id
            self.colocal_ids[clc_id] = {k:v for k,v in ch_dict.items() if k!="colocal_id"}
        
        # parse colocal nuclei info 
        if self.colocal_nuclei_info is not None:
            for clc_dict in self.colocal_nuclei_info:
                # get colocal chs and use these to get the colocal_ids
                coChs = self.check_not_none(clc_dict.get("ch_idx"))
                if len(coChs) < 2:
                    raise ValueError (f"colocalization must be between >2 channels, got {coChs}")
                coIds = [self.get_colocal_id_from_ch_idx(ch) for ch in coChs] #(clc_dict.get("co_ids"))
                assert len(coChs) == len(coIds)
                
                # ensure assigned_colocal_id is unique
                assign_colocal_id = self.check_not_none(clc_dict.get("colocal_id"))
                if assign_colocal_id in self.colocal_ids:
                    raise ValueError (f"colocal_id must be unique, got {assign_colocal_id} but have {self.colocal_ids}")
                
                # TODO too much redundancy here, perhaps can cut down?
                self.colocal_ids[assign_colocal_id] = {k:v for k,v in clc_dict.items() if k!="colocal_id"}
                self.colocalizations.append({
                    "name":clc_dict["name"], "coChs":coChs, "coIds":coIds, 'assign_colocal_id':assign_colocal_id,
                    "intersecting_label_columns":[f"ch{ch}_intersecting_label" for ch in coChs[:-1]],
                    "intersecting_colocal_ids":coIds[:-1],
                    "base_colocal_id":coIds[-1],
                })
                self.colocalized_ids.append(assign_colocal_id)

        self.sort_colocal_ids()
        
    def __str__(self):
        astr = ''
        for k,v in self.__dict__.items():
            
            _str = f"{k}: {v}\n"
            if isinstance(v, list):
                if len(v) == 0:
                    _str = f'{k}: {v}\n'
                elif isinstance(v[0], dict):
                    _str = f'{k}:\n'
                    for d in v:
                        _str += f'\t{d}\n'
            elif isinstance(v, dict):
                if isinstance(v[list(v.keys())[0]], dict):
                    _str = f'{k}:\n'
                    for kk, vv in v.items():
                        _str += f'\t{kk}: {vv}\n'
            astr += _str
        return astr

    def __repr__(self):
        return self.__str__()
            
    def sort_colocal_ids(self):
        setattr(self, 'colocal_ids', {k: self.colocal_ids[k] for k in sorted(self.colocal_ids)})

    def reformat_json_params(self, param_attr):
        # reformat param dicts so keys are int not strs (.json requires them to be strings)
        old_params, new_params = getattr(self, param_attr), {}
        for cohort, param_dicts in old_params.items():
            new_params[cohort] = {}
            for ch, param_dict in param_dicts.items():
                new_params[cohort][int(ch)] = deepcopy(param_dict)
        delattr(self, param_attr)
        setattr(self, param_attr, new_params)
            
    def get_colocal_id_from_ch_idx(self, ch_idx):
        val = self.colocalid_ch_map.get(ch_idx)
        if val is not None:
            return val
        raise KeyError(f"cannot find {ch_idx} in colocalid_ch_map")
    

    def check_not_none(self, var):
        if var is None:
            raise KeyError(f"{var} is None, ensure it is properly defined in config file.")
        return var

    def check_exists(self, attr):
        if not hasattr(self, attr):
            raise KeyError(f"{attr} not found, ensure it is defined in config file.")
        elif getattr(self, attr) is None:
            raise KeyError(f"{attr} is None, ensure it is defined in config file.")
        else:
            return 0  # all good
        
    def get_colocalid_ch_map(self):
        """"returns dict mapping channels in intensity image to colocal id"""
        self.check_exists("colocalid_ch_map")
        return self.colocalid_ch_map
    
    def get_count_channel_names(self):
        self.sort_colocal_ids()
        return [f"n{self.colocal_ids[k]['name']}" for k in self.colocal_ids]
     
    def get_clc_nuc_info(self): # TODO remove '_nuc'
        """ returns dict mapping self.colocalized_ids to colocalization info """
        clc_nuc_info = {}
        for coloc in self.colocalizations:
            clc_nuc_info[coloc['assign_colocal_id']] = {k:v for k,v in coloc.items() if k != 'assign_colocal_id'}
        return clc_nuc_info

    def get_colocalized_ids(self):
        """ returns ids that assigned in self.colocalizations """
        return self.colocalized_ids
    
    def get_colocalization_info(self, colocal_id):
        """ returns dict if colocal_id defines a colocalized population (e.g. is one of self.colocalized_ids) """
        for clz_info in self.colocalizations:
            if colocal_id == clz_info['assign_colocal_id']:
                return clz_info
        raise ValueError(f"{colocal_id} not in {self.colocalizations}, id must match one of the assign_colocal_id")

    def get_shared_base_ids(self, colocal_id):
        """ get ids in colocalizations that share the same base_colocal_id as this one """

        this_id_info = self.get_colocalization_info(colocal_id)
        this_base_id = this_id_info['base_colocal_id']
        
        shared_base_ids = {}
        for clz_info in self.colocalizations:
            if (clz_info['assign_colocal_id'] != colocal_id) and (clz_info['base_colocal_id'] == this_base_id):
                shared_base_ids[clz_info['assign_colocal_id']] = clz_info['base_colocal_id']
        
        return shared_base_ids
    
                    
    def get_inherited_colocalizations(self, colocal_id):
        """ get ids of all colocalizations that this id encompasses 
                i.e. explicitly it's base_id and implicitly those that share a base_colocal_id and contain all of the same coIds
                e.g. in the doc string example, clc 4 inherits from 2 as well as 3 (the colocalization of 1 and 2)
        """
        clz_info = self.get_colocalization_info(colocal_id)
        
        implicit_ids = list(self.get_shared_base_ids(colocal_id).keys())

        # check if all coIds are shared
        def allin(list1, list2):
            """checks if all elements in list1 are in list2"""
            return all([i in list2 for i in list1])
        
        implicit_ids = [i for i in implicit_ids if allin(self.get_colocalization_info(i)['coIds'], clz_info['coIds'])]
        return [clz_info['base_colocal_id']] + implicit_ids
    
    
    def get_parents(self, base_clc_ids: int | list):
        """ return all ids that contain any of the base_clc_id(s) """
        
        base_clc_ids = [base_clc_ids] if not isinstance(base_clc_ids, list) else base_clc_ids

        return list({
            d['assign_colocal_id']
            for d in self.colocalizations
            if any(clci in d['coIds'] for clci in base_clc_ids)
        })


