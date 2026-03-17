"""
############################################################################################################
    DESCRIPTION
    ~~~~~~~~~~~
        Helper functions and classes to load and query brain atlas ontology data.
        Supports multiple ontology formats (ABBA v3p1, CCFv3, Kim mouse atlas).
    
    CORE CLASSES
    ~~~~~~~~~~~~
        - Ontology: Main class for loading and querying ontology data
        - OntologyIndexer: Provides convenient indexing by structural level
        - IDList: Enhanced list with methods for querying atlas properties
    
    ONTOLOGY STRUCTURE LEVELS
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    - regions by st_level for Basic cell groups and regions:
        1--> nRegions: 1    |  8--> nRegions: 298   
        2--> nRegions: 3    |  9--> nRegions: 177
        3--> nRegions: 4    |  10--> nRegions: 46
        4--> nRegions: 1    |  11--> nRegions: 507
        5--> nRegions: 13   |
        6--> nRegions: 34   |
        7--> nRegions: 19   |
############################################################################################################
"""
import json
from collections import deque
import pandas as pd
import numpy as np
from typing import Any, Optional


# Default ontology path
CCFV3_JSON_ONTOLOGY_PATH = r""


class IDList(list):
    """
    Enhanced list subclass for atlas region IDs with built-in query methods.
    Maintains a reference to the parent OntologyIndexer for attribute lookups.
    """
    
    def __init__(self, ids, indexer):
        super().__init__(ids)
        self._indexer = indexer

    def _get_attribute(self, attr):
        """Get attribute values for all IDs in this list"""
        return self._indexer.ont.get_attributes_for_list_of_ids(self, attr)
    
    def parents(self) -> "IDList":
        """Get parent structure IDs"""
        pars = self._get_attribute('parent_structure_id')
        return IDList(pars, self._indexer)
    
    def names(self) -> list[str]:
        """Get region names"""
        return self._get_attribute('name')

    def acronyms(self) -> list[str]:
        """Get region acronyms"""
        return self._get_attribute('acronym')
    
    def st_levels(self) -> list[int]:
        """Get structural levels"""
        return self._get_attribute('st_level')

    def children(self, flat=False) -> Any: # -> list["IDList"] | "IDList":
        """ Get direct children for each ID """
        children_list = []
        for id_val in self:
            children = self._indexer.ont.get_children(id_val)
            if flat:
                children_list.extend(children)
            else:
                children_list.append(IDList(children, self._indexer))
        return IDList(children_list, self._indexer) if flat else children_list 

    def all_children(self) -> "IDList":
        """Get all descendant IDs (recursive)"""
        all_children = []
        for id_val in self:
            children = self._indexer.ont.get_all_children(id_val)
            all_children.extend([id_val] + children)
        return IDList(self._remove_duplicates(all_children), self._indexer)
    
    def parent_at_lvl(self, at_st_level: int):
        """ get the parent region id at a specific st_level """     
        parent_at_st = []
        for id_val in self:
            
            curr_st = self._indexer(id_val).st_levels()[0]
            
            if curr_st < at_st_level: # undefined if a region is at a lower st_level than requested
                parent_at_st.append(np.nan)
            elif curr_st == at_st_level: # if this is a parent at desired st_level
                parent_at_st.append(id_val)
            else:
                # recursively check st_level of regions up parent heirarchy
                while True:
                    parent_id = self._indexer(id_val).parents()[0]
                    parst = self._indexer(parent_id).st_levels()[0]
                    if parst == at_st_level:
                        parent_at_st.append(parent_id)
                        break
                    id_val = parent_id

        return IDList(parent_at_st, self._indexer)

    def unique_parents(self, print_summary=True):
        """
        Group regions by their parent, returning mapping of parent_id -> child_ids
        """
        parent_groups = {}
        for reg_id in self:
            parent_id = self._indexer.ont.ont_ids[reg_id]["parent_structure_id"]
            if parent_id not in parent_groups:
                parent_groups[parent_id] = set()
            
            # Add all children of this parent
            for child in self._indexer.ont.ont_ids[parent_id]["children"]:
                parent_groups[parent_id].add(child["id"])
        
        if print_summary:
            for parent_id, child_ids in parent_groups.items():
                parent_info = self._indexer.ont.ont_ids[parent_id]
                child_names = self._indexer.ont.get_attributes_for_list_of_ids(
                    list(child_ids), 'name'
                )
                print(f"st_level: {parent_info['st_level']} parent_id:{parent_id} "
                      f"({parent_info['name']}): {list(child_ids)}, {child_names}")
        
        return parent_groups
    
    def flat(self):
        """ removes nesting, returns IDlist if possible """
        return IDList(flatten_list(self), self._indexer)

    @staticmethod
    def _remove_duplicates(lst):
        """Remove duplicates while preserving order"""
        seen = set()
        result = []
        for item in lst:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result


class OntologyIndexer:
    """
    Provides convenient indexing and querying interface for ontology data.
    Supports indexing by structural level: indexer.st(5) or indexer.st(5, 6, 8)
    """
    
    def __init__(self, ontology):
        self.ont = ontology
        self.ids_by_st_level = self._organize_ids_by_st_level()
    
    def _organize_ids_by_st_level(self):
        """Organize all IDs by their structural level"""
        st_level_dict = {}
        for st_level, ids in self._gather_ids_by_st_level(as_dict=True).items():
            # Remove duplicates while preserving order
            unique_ids = IDList._remove_duplicates(ids)
            st_level_dict[st_level] = unique_ids
        return st_level_dict
    
    def _gather_ids_by_st_level(self, as_dict=False):
        """Recursively gather all IDs organized by structural level"""
        ids_and_levels = []
        
        def _traverse(data):
            if isinstance(data, dict):
                if "id" in data and "st_level" in data:
                    ids_and_levels.append((data["st_level"], data["id"]))
                for value in data.values():
                    _traverse(value)
            elif isinstance(data, list):
                for item in data:
                    _traverse(item)
        
        _traverse(self.ont.ont_ids)
        
        if as_dict:
            result = {}
            for st_level, id_val in ids_and_levels:
                if st_level not in result:
                    result[st_level] = []
                result[st_level].append(id_val)
            return result
        return ids_and_levels

    def __call__(self, ids):
        """Convert ID or list of IDs to IDList object"""
        if isinstance(ids, (int, float)):
            ids = [int(ids)]
        elif not isinstance(ids, list):
            raise TypeError(f"Expected int, float, or list, got {type(ids)}")
        return IDList([int(id_val) for id_val in ids], self)

    def st(self, *levels):
        """
        Get regions at specified structural level(s).
        Usage: indexer.st(5) or indexer.st(5, 6, 8)
        """
        if len(levels) == 1:
            level = levels[0]
            if level not in self.ids_by_st_level:
                raise KeyError(f"Structural level {level} not found")
            return IDList(self.ids_by_st_level[level], self)
        
        # Multiple levels - merge results
        merged_ids = []
        for level in levels:
            if level not in self.ids_by_st_level:
                raise KeyError(f"Structural level {level} not found")
            merged_ids.extend(self.ids_by_st_level[level])
        
        return IDList(IDList._remove_duplicates(merged_ids), self)


class Ontology:
    """
    Main class for loading and querying brain atlas ontology data.
    Supports multiple ontology formats and provides comprehensive querying capabilities.
    """
    
    def __init__(self, source: str | pd.DataFrame = CCFV3_JSON_ONTOLOGY_PATH):
        """
        Initialize ontology from JSON file or DataFrame.
        
        Args:
            source: Path to JSON file or pandas DataFrame with ontology data
        """
        if isinstance(source, str): 
            self.ont_ids = self._load_from_json(source)
        elif isinstance(source, pd.DataFrame):
            self.ont_ids = self._load_from_dataframe(source)
        else:
            raise TypeError("Source must be JSON file path or pandas DataFrame")
        
        # Create name -> ID mapping for convenient lookups
        self.names_dict = {
            data["name"]: id_val for id_val, data in self.ont_ids.items()
        }
        
        self.ONI = OntologyIndexer(self)
        
        # Color mapping for major brain regions (st_level 5)
        self.parent_level_colormap = {
            "Cortical subplate": "#3288bd",
            "Isocortex": "#66c2a5", 
            "Olfactory areas": "#abdda4",
            "Pallidum": "#e6f598",
            "Hippocampal formation": "#ffffbf",
            "Striatum": "#fee08b",
            "Thalamus": "#fdae61",
            "Hypothalamus": "#f46d43",
            "Midbrain": "#d53e4f",
            "Pons": "#980043",
            "Medulla": "#756bb1",
            "Cerebellar cortex": "#d9d9d9",
            "Cerebellar nuclei": "#969696",
        }
    
    def _load_from_json(self, json_path):
        """Load ontology from JSON file"""
        with open(json_path, "r", encoding="latin-1") as f:
            data = json.load(f)
        
        print('warning: json source may not be fully supported by new implementation if missing keys like parent_structure_id')
        # TODO: when loading e.g. the ontology.json output by abba export need to build parent structure path from scratch
        # these ontology are missing the following keys, some are used in new implementation
        # 'parent_structure_id', 'st_level', 'atlas_id', 'graph_order', 'hemisphere_id', 
        
        # Handle different ontology formats
        if "root" in data.keys():
            all_groups = data["root"]["children"]
        elif "msg" in data.keys():
            all_groups = data["msg"][0]["children"]
        else:
            raise KeyError(f"data must contain root or msg key but has: {data.keys()}")
        
        # Handle new vs old ontology structure
        if all_groups and 'data' in all_groups[0]:
            all_groups = self._flatten_nested_data(all_groups)
        
        return self._build_id_dict(all_groups)
    
    def _load_from_dataframe(self, df):
        """Load ontology from pandas DataFrame"""
        df = self._prep_dataframe(df)
        tree = self._build_tree_from_df(df.sort_values('st_level'))
        all_groups = self._flatten_nested_data(tree['children'])
        return self._build_id_dict(all_groups)
    
    def _flatten_nested_data(self, data):
        """Convert new ontology format (data nested) to flat structure"""
        if isinstance(data, dict):
            flattened = {k: v for k, v in data.items() if k != 'data'}
            
            if 'data' in data and isinstance(data['data'], dict):
                for key, value in data['data'].items():
                    if key in ['id', 'parent_structure_id', 'atlas_id', 
                              'graph_order', 'st_level', 'hemisphere_id']:
                        value = int(value)
                    flattened[key] = value
            
            if 'children' in data and isinstance(data['children'], list):
                flattened['children'] = [
                    self._flatten_nested_data(child) for child in data['children']
                ]
            
            return flattened
        
        elif isinstance(data, list):
            return [self._flatten_nested_data(item) for item in data]
        
        return data
    
    def _build_id_dict(self, nested_data):
        """Convert nested ontology structure to flat ID -> attributes dictionary"""
        result = {}
        
        def _traverse(items):
            for item in items:
                result[item["id"]] = item
                if "children" in item and item["children"]:
                    _traverse(item["children"])
        
        _traverse(nested_data)
        return result
    
    def _prep_dataframe(self, df):
        """Prepare DataFrame for ontology construction"""
        df = df.copy()
        df['id'] = df['id'].astype('str')
        df["structure_paths"] = (df["structure_id_path"]
                                .str.strip('/').str.strip('\\')
                                .str.split('/'))
        df["structure_paths"] = df["structure_paths"].apply(
            lambda x: [str(int(el)) for el in x]
        )
        df["st_level"] = df["structure_paths"].apply(len)
        df["parent_structure_id"] = (df["parent_structure_id"]
                                    .fillna(0).astype('int').astype('str'))
        
        # Add missing columns
        for col in ['atlas_id', 'graph_order', 'hemisphere_id']:
            if col not in df.columns:
                df[col] = np.nan
        
        return df.fillna(-1)
    
    def _build_tree_from_df(self, df):
        """Build hierarchical tree structure from DataFrame"""
        tree = {
            'id': 997,
            'color': [255, 255, 255, 255],
            'data': {'name': 'root', 'acronym': 'root', 'id': '997'},
            'children': []
        }
        
        def find_child(children, child_id):
            for child in children:
                if child['id'] == child_id:
                    return child
            return None
        
        for _, row in df.iterrows():
            current_node = tree
            
            for struct_id in row['structure_paths']:
                child_node = find_child(current_node['children'], struct_id)
                
                if not child_node:
                    node_data = df[df['id'] == struct_id]
                    if not node_data.empty:
                        node_row = node_data.iloc[0]
                        data_dict = {
                            'name': node_row['name'].strip('"'),
                            'acronym': node_row['acronym'],
                            'id': str(node_row['id']),
                            'parent_structure_id': node_row['parent_structure_id'],
                            'atlas_id': node_row['atlas_id'],
                            'graph_order': node_row['graph_order'],
                            'hemisphere_id': int(node_row['hemisphere_id']),
                            'st_level': int(node_row['st_level']),
                        }
                    else:
                        data_dict = {
                            'name': f'unknown_{struct_id}',
                            'acronym': f'unk_{struct_id}',
                            'id': str(struct_id),
                            'parent_structure_id': -1,
                            'atlas_id': -1,
                            'graph_order': -1,
                            'hemisphere_id': -1,
                            'st_level': -1,
                        }
                    
                    child_node = {
                        'id': struct_id,
                        'color': [255, 255, 255, 255],
                        'data': data_dict,
                        'children': []
                    }
                    current_node['children'].append(child_node)
                
                current_node = child_node
        
        return tree['children'][0]
    
    def get_indexer(self):
        """Get an OntologyIndexer for convenient querying"""
        return self.ONI
    
    def get_children(self, parent_id):
        """ Get direct children ids """
        return [child['id'] for child in self.ont_ids[parent_id]['children']]
    
    def get_all_children(self, parent_id):
        """Get all descendant IDs for a given parent ID (recursive)"""
        if parent_id not in self.ont_ids:
            return []
        
        children = []
        queue = deque(self.ont_ids[parent_id].get("children", []))
        
        while queue:
            child = queue.popleft()
            if isinstance(child, dict) and "id" in child:
                children.append(child["id"])
                queue.extend(child.get("children", []))
        
        return children
    
    def get_all_parents(self, region_id, max_region_id=997):
        """Get all ancestor IDs for a given region (up to root)"""
        parents = []
        current_id = region_id
        
        while current_id != max_region_id and current_id is not None:
            if current_id not in self.ont_ids:
                break
            parent_id = self.ont_ids[current_id].get("parent_structure_id")
            if parent_id is None:
                break
            parents.append(parent_id)
            current_id = parent_id
        
        return parents
    
    def get_attributes_for_list_of_ids(self, id_list, attribute, warn=False, missing_value="notFound"):
        """Get specified attribute for a list of region IDs"""
        results = []
        missing_ids = []
        
        for region_id in id_list:
            try:
                if region_id not in self.ont_ids:
                    raise KeyError(f"ID {region_id} not found")
                
                data = self.ont_ids[region_id]
                # Handle nested data structure
                if "data" in data and attribute != "children":
                    data = data["data"]
                
                results.append(data[attribute])
            except KeyError:
                missing_ids.append(region_id)
                results.append(missing_value)
        
        if missing_ids and warn:
            unique_missing = set(missing_ids)
            print(f"WARNING: {len(unique_missing)} unique atlas IDs not found: "
                  f"{unique_missing} (total: {len(missing_ids)})")
        
        return results
    
    def to_dataframe(self, include_attributes=None):
        """Convert ontology to pandas DataFrame"""
        if include_attributes is None:
            include_attributes = ['acronym', 'parent_structure_id', 'st_level']
        
        rows = []
        for region_id, data in self.ont_ids.items():
            row = {
                'region_id': region_id,
                'region_name': data['name'],
                'acronym': data['acronym']
            }
            
            for attr in include_attributes:
                row[attr] = data.get(attr, None)
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def print_structure_summary(self, return_dict=False):
        """Print summary of regions by structural level"""
        indexer = self.get_indexer()
        st_level_dict = indexer.ids_by_st_level
        
        for st_level in sorted(st_level_dict.keys()):
            ids = st_level_dict[st_level]
            names = self.get_attributes_for_list_of_ids(ids, "name")
            print(f"st_level {st_level}: {len(ids)} regions")
            print(f"  IDs: {ids}")
            print(f"  Names: {names}")
            print()
        
        if return_dict:
            return st_level_dict
    
    def get_regions_at_level(self, st_level, max_st_level=None):
        """Get all region names at specified structural level(s)"""
        indexer = self.get_indexer()
        
        if max_st_level is None:
            max_st_level = 99
        
        region_ids = []
        for level, ids in indexer.ids_by_st_level.items():
            if level == st_level or (isinstance(st_level, (list, tuple)) and level in st_level):
                if level <= max_st_level:
                    region_ids.extend(ids)
        
        # Remove duplicates while preserving order
        unique_ids = IDList._remove_duplicates(region_ids)
        return self.get_attributes_for_list_of_ids(unique_ids, "name")
    
    def map_children_to_parents(self, parent_st_level=5):
        """Create mapping from child region names to parent region names at specified level"""
        indexer = self.get_indexer()
        parent_regions = indexer.st(parent_st_level)
        
        child_to_parent = {}
        for parent_id in parent_regions:
            parent_name = self.ont_ids[parent_id]["name"]
            child_ids = self.get_all_children(parent_id)
            child_names = self.get_attributes_for_list_of_ids(child_ids, "name")
            
            for child_name in child_names:
                child_to_parent[child_name] = parent_name
        
        return child_to_parent


# Utility functions (kept for backwards compatibility)
def load_ontology(json_path=CCFV3_JSON_ONTOLOGY_PATH):
    """Load ontology - wrapper for backwards compatibility"""
    ont = Ontology(json_path)
    return ont.ont_ids

def print_possible_attributes():
    """Print example of possible region attributes"""
    example = {
        "id": 997,
        "atlas_id": -1,
        "ontology_id": 1,
        "acronym": "root",
        "name": "root",
        "color_hex_triplet": "FFFFFF",
        "graph_order": 0,
        "st_level": 0,
        "hemisphere_id": 3,
        "parent_structure_id": None,
        "children": [],
    }
    print(json.dumps(example, indent=2))

def flatten_list(nested_list):
    """recursively flatten lists of sublists"""
    flat_list = []
    for item in nested_list:
        if isinstance(item, (list, IDList)):
            flat_list.extend(flatten_list(item))  # Recursively flatten if it's a list
        else:
            flat_list.append(item)
    return flat_list

# version replaced 6/4/25 for clean-up, above has yet to be significantly tested
# import json
# from collections import deque
# import pandas as pd
# import numpy as np

# """
# ############################################################################################################
#     DESCRIPTION
#     ~~~~~~~~~~~
#         helper functions to load an ontology json file, to extract atlas ids and the various properties
    
#     NOTES
#     ~~~~~
#         5/29 - updated load_ontology and extract_ont_label_key to allow parsing of abba v3p1 ontology, 
#             organization is different as all attrs we care about except children list are contained 
#             in attribute named data instead of just keys in the dict
    
#     ONTOLOGY DESCTIPTIONS
#     ~~~~~~~~~~~~~~~~~~~~~
#     - regions by st_level for Basic cell groups and regions, much higher when including all 4 st_level 1's
#         1--> nRegions: 1
#         2--> nRegions: 3
#         3--> nRegions: 4
#         4--> nRegions: 1
#         5--> nRegions: 13
#         6--> nRegions: 34
#         7--> nRegions: 19
#         8--> nRegions: 298
#         9--> nRegions: 177
#         10--> nRegions: 46
#         11--> nRegions: 507
# ############################################################################################################
# """

# # CCFV3_JSON_ONTOLOGY_PATH = r"F:\ABBA\abba_atlases\1.json"
# CCFV3_JSON_ONTOLOGY_PATH = r"R:\Confocal data archive\Pascal\ANALYSIS\ABBA_synapse_quant_testing\abba_syn_quant_test_qupath_project\Adult Mouse Brain - Allen Brain Atlas V3p1-Ontology.json"

# def load_ontology(json_path=CCFV3_JSON_ONTOLOGY_PATH):
#     """
#     load a list of dicts where keys are atlas_ids and values are dict of all properties and children
#         read the 1.json file in abba_atlases                OR
#         read the v3p1-ontology created during abba export
#             (though not all functions may work as org is different, e.g. attrs in data attribute)
#     """
#     # also support generation via dict object directly
#     if isinstance(json_path, str):
#         # read the json file
#         with open(json_path, "r") as f:
#             img_labels_json = json.load(f)


#         # handle the different ontologies, which have varying organizations of the attributes
#         if json_path.endswith("1.json"):
#             all_groups = img_labels_json["msg"][0]["children"]
#         elif json_path.endswith("Adult Mouse Brain - Allen Brain Atlas V3p1-Ontology.json"):
#             all_groups = img_labels_json["root"]["children"]
#         elif json_path.endswith("kim_mouse_10um-Ontology.json"):
#             all_groups = img_labels_json["root"]["children"]
#         else:
#             # general case
#             all_groups = img_labels_json["msg"][0]["children"]
#     else:
#         assert isinstance(json_path, dict)
#         all_groups = json_path

#     # handle differences in ontology structure introduced in abba version changes
#     old_version = False if ('data' in all_groups[0]) else True
#     if not old_version:
#         all_groups = recreate_structure_with_flat_data(all_groups)

    
#     # convert nested ontology to dict of id:attributes
#     ids = iterate_dicts(all_groups)
#     return ids


# def recreate_structure_with_flat_data(data):
#     """
#     convert new ontology to old version where attributes were not nested inside the 'data' key
#     returns dictionary where keys are atlas ids and values are attributes of that region

#     Recursively visits each dictionary in the 'children' key,
#     flattens the 'data' dictionary into the main dictionary,
#     and recreates the hierarchical structure.
    
#     Parameters:
#         data (dict or list): The input data, which can be a dictionary or a list of dictionaries.
    
#     Returns:
#         dict or list: A new structure with the 'data' key flattened.
#     """
#     if isinstance(data, dict):
#         # Create a new dictionary with 'data' flattened
#         flattened_dict = {k: v for k, v in data.items() if k != 'data'}
#         if 'data' in data and isinstance(data['data'], dict):
#             # Pull out each key-value pair from 'data' and add it to the main dictionary
#             for k,v in data['data'].items():
#                 if k in ['id', 'parent_structure_id', 'atlas_id', 'graph_order', 'st_level', 'hemisphere_id']: 
#                     v = int(v)
#                 flattened_dict[k] = v
        
#         # If 'children' key is present, process each child recursively
#         if 'children' in data and isinstance(data['children'], list):
#             flattened_dict['children'] = [
#                 recreate_structure_with_flat_data(child) for child in data['children']
#             ]
        
#         return flattened_dict
#     elif isinstance(data, list):
#         # If the data is a list, process each element recursively
#         return [recreate_structure_with_flat_data(item) for item in data]
#     else:
#         return data
    
    

# def ont_from_structsdf(df):
#     df = prep_structs_df(df)
#     tree = build_tree(df.sort_values('st_level'))
#     return tree


# def build_tree(df):
#     """ structs df to ont tree """
#     # data cols: 
#     data_attributes = ['id', 'parent_structure_id', 'atlas_id', 'graph_order', 'st_level', 'hemisphere_id']

#     # Root node
#     tree = {
#         'id': 997,
#         'color': [255, 255, 255, 255],
#         'data': {'name': 'root', 'acronym': 'root', 'id': '997'},
#         'children': []
#     }

#     def find_child(children, cid):
#         """Find a child node by ID in a list of children."""
#         for child in children:
#             if child['id'] == cid:
#                 return child
#         return None

#     for _, row in df.iterrows():
#         path = row['spaths']
#         current_node = tree

#         for struct_id in path:
#             # Try to find this ID in the current node's children
#             child_node = find_child(current_node['children'], struct_id)

#             if not child_node:
#                 # Try to get data from the DataFrame if available
#                 node_data = df[df['id'] == struct_id]
#                 if not node_data.empty:
#                     node_data = node_data.iloc[0]
#                     data_d = dict(
#                         name = node_data['name'].strip('"'),
#                         acronym = node_data['acronym'],
#                         id = str(node_data['id']),
#                         parent_structure_id = node_data['parent_structure_id'],
#                         atlas_id = node_data['atlas_id'],
#                         graph_order = node_data['graph_order'],
#                         hemisphere_id = int(node_data['hemisphere_id']),
#                         st_level = int(node_data['st_level']),
#                     )
#                 else:
#                     data_d = dict(zip(data_attributes, [-1]*data_attributes))
#                     data_d['id'] = str(struct_id)

#                 child_node = {
#                     'id': struct_id,
#                     'color': [255, 255, 255, 255],
#                     'data': data_d,
#                     'children': []
#                 }
#                 current_node['children'].append(child_node)

#             # Descend to this node for next iteration
#             current_node = child_node

#     return tree['children'][0]



# def prep_structs_df(df):
#     """ calculate st_level by counting the number of structures in the structure_id_path"""
#     import numpy as np
#     df['id'] = df['id'].astype('str')
#     df["spaths"] = df["structure_id_path"].str.strip(r'/').str.strip('\\').str.split(r'/')
#     df["spaths"] = df["spaths"].apply(lambda x: [str(int(el)) for el in x])
#     df["st_level"] = df["spaths"].apply(lambda x: len(x))
#     df["parent_structure_id"] = df["parent_structure_id"].fillna(0).astype('int').astype('str')

#     # add columns not present
#     _has = ['id', 'parent_structure_id', 'st_level'] # should already contain these 
#     cols = ['atlas_id', 'graph_order', 'hemisphere_id']
#     for c in cols:
#         if c not in df:
#             df[c] = np.nan
#     df = df.fillna(-1)
#     return df



# class Ontology:
#     def __init__(self, json_path=CCFV3_JSON_ONTOLOGY_PATH):
#         if isinstance(json_path, str):
#             self.ont_ids = load_ontology(json_path)
#         else:
#             assert isinstance(json_path, pd.DataFrame)
#             self.df = prep_structs_df(json_path)
#             self.tree = build_tree(self.df.sort_values('st_level'))['children']
#             all_groups = recreate_structure_with_flat_data(self.tree)
#             self.ont_ids = iterate_dicts(all_groups)
            

#         self.names_dict = dict(
#             zip([d["name"] for d in self.ont_ids.values()], self.ont_ids.keys())
#         )

#         # st5 color map
#         self.parent_level_colormap = {
#             "Cortical subplate": "#3288bd",
#             "Isocortex": "#66c2a5",
#             "Olfactory areas": "#abdda4",
#             "Pallidum": "#e6f598",
#             "Hippocampal formation": "#ffffbf",
#             "Striatum": "#fee08b",
#             "Thalamus": "#fdae61",
#             "Hypothalamus": "#f46d43",
#             "Midbrain": "#d53e4f",
#             "Pons": "#980043",
#             "Medulla": "#756bb1",
#             "Cerebellar cortex": "#d9d9d9",
#             "Cerebellar nuclei": "#969696",
#         }

#     def map_region_to_parent_st_level(self, region_names):
#         """not implemented"""
#         output = {}  # dict mapping parent at a specific st_level to children region names
#         for reg in region_names:
#             pass

#     def get_all_children(self, parent_id):
#         """returns a list of all children ids for a given parent_id, by parsing the nested list of its children key"""
#         nested_list = self.ont_ids[parent_id]["children"]
#         ids = []

#         def helper(nested_item):
#             if isinstance(nested_item, list):
#                 for item in nested_item:
#                     helper(item)
#             elif isinstance(nested_item, dict):
#                 if "id" in nested_item:
#                     ids.append(nested_item["id"])
#                 if "children" in nested_item:
#                     helper(nested_item["children"])

#         assert isinstance(nested_list, list), "please pass a list"
#         helper(nested_list)
#         return ids

#     def get_attributes_for_list_of_ids(
#         self, list_of_ids, get_key, warn=False, nan_val="notFound"
#     ):
#         """though some atlas regions do not appear in ontology they seem to be insignificant"""
#         ontology_by_ids = self.ont_ids
#         extracted_attributes = []
#         atlas_region_lookup_error_ids = []
#         for x in list_of_ids:
#             try:
#                 val = extract_ont_label_key(ontology_by_ids, x, get_key)
#             except KeyError:
#                 atlas_region_lookup_error_ids.append(x)
#                 val = nan_val
#             extracted_attributes.append(val)
#         lookup_error_ids_unique = set(atlas_region_lookup_error_ids)
#         if atlas_region_lookup_error_ids and warn:
#             print(
#                 f"WARN n unique atlas ids not found: {len(lookup_error_ids_unique)} ({lookup_error_ids_unique}) total: ({len(atlas_region_lookup_error_ids)})"
#             )
#         return extracted_attributes


# def print_possible_attributes():
#     """prints the possible attributes of a atlas region as stored in the ontology"""
#     print(
#         {
#             "id": 997,
#             "atlas_id": -1,
#             "ontology_id": 1,
#             "acronym": "root",
#             "name": "root",
#             "color_hex_triplet": "FFFFFF",
#             "graph_order": 0,
#             "st_level": 0,
#             "hemisphere_id": 3,
#             "parent_structure_id": None,
#             "children": [],
#         }
#     )


# def print_regions_by_st_level(ont_ids, return_dict=False):
#     """# can also filter by a specific id level e.g. ont_ids[8]
#     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     # example input: ont.ont_ids[8], output:
#     # 1--> nRegions: 1
#     #     [8]
#     #     ['Basic cell groups and regions']
#     # 2--> nRegions: 3
#     #     [567, 343, 512]
#     #     ['Cerebrum', 'Brain stem', 'Cerebellum']
#     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
#     ont = Ontology()
#     st_order_dict = gather_ids_by_st_level(ont_ids, as_dict=True)
#     for st_level, v in sorted(st_order_dict.items(), key=lambda x: int(x[0])):
#         print(
#             f"st_level:{st_level}--> nRegions: {len(v)}",
#             "\n\t",
#             v,
#             "\n\t",
#             get_attributes_for_list_of_ids(ont.ont_ids, v, "name"),
#         )
#     if return_dict:
#         return st_order_dict


# def get_unique_parents(reg_ids, p=True):
#     """for each id get its parent, and return dict mapping unique parents to list of ids from input list that fall under that parrent
#     example usage:
#     reg_ids: [315, 698, 1089, 703, 477, 803, 549, 1097, 313, 771, 354, 528, 519]
#     output:
#         st_lvl: 4 parent_id:695 (Cortical plate): [1089, 698, 315], ['Hippocampal formation', 'Olfactory areas', 'Isocortex']
#         st_lvl: 3 parent_id:688 (Cerebral cortex): [703, 695], ['Cortical subplate', 'Cortical plate']
#         st_lvl: 3 parent_id:623 (Cerebral nuclei): [803, 477], ['Pallidum', 'Striatum']
#         st_lvl: 3 parent_id:1129 (Interbrain): [1097, 549], ['Hypothalamus', 'Thalamus']
#         st_lvl: 2 parent_id:343 (Brain stem): [1129, 1065, 313], ['Interbrain', 'Hindbrain', 'Midbrain']
#         st_lvl: 3 parent_id:1065 (Hindbrain): [354, 771], ['Medulla', 'Pons']
#         st_lvl: 2 parent_id:512 (Cerebellum): [528, 519], ['Cerebellar cortex', 'Cerebellar nuclei']
#         d = {695: {315, 698, 1089},
#             688: {695, 703},
#             623: {477, 803},
#             1129: {549, 1097},
#             343: {313, 1065, 1129},
#             1065: {354, 771},
#             512: {519, 528}}
#     """
#     ont = Ontology()
#     d = {}
#     for reg_id in reg_ids:
#         parent_id = ont.ont_ids[reg_id]["parent_structure_id"]
#         if parent_id not in d:
#             d[parent_id] = set()
#         for c in ont.ont_ids[parent_id]["children"]:
#             d[parent_id].add(c["id"])
#     for k, v in d.items():  # parent_id, list of child ids
#         if p: print(
#             f"st_lvl: {ont.ont_ids[k]['st_level']} parent_id:{k} ({ont.ont_ids[k]['name']}): {list(v)}, {get_attributes_for_list_of_ids(ont.ont_ids, v, 'name')}"
#         )
#     return d


# def ont2df(ont_ids, extract_attrs=['acronym', 'parent_structure_id', 'st_level']):
#     import pandas as pd
#     names_dict = dict(zip([d["name"] for d in ont_ids.values()], ont_ids.keys()))
#     extract_attr_fxn = lambda ontids, rid, attrs: {a:ontids[rid][a] for a in attrs}
    
#     ont_df = []
#     for k, v in names_dict.items():
#         ont_df.append(
#             {"region_id": v, "region_name": k, "acronym": ont_ids[v]["acronym"], **extract_attr_fxn(ont_ids, v, extract_attrs)}
#         )
#     ont_df = pd.DataFrame(ont_df)
#     # ont_df.to_excel('ontology_dataframe.xlsx')
#     return ont_df


# def load_lowest_structural_level_only(json_path):
#     raw_ontology = load_raw_ontology(json_path)
#     result = find_empty_children(raw_ontology)
#     return result


# def load_raw_ontology(json_path=CCFV3_JSON_ONTOLOGY_PATH):
#     with open(json_path, "r") as f:
#         img_labels_json = json.load(f)
#     return img_labels_json["msg"][0]["children"]





# def iterate_dicts(dicts, result=None, counter=0):
#     """convert nested ontology to dictionary where keys are atlas ids and values are attributes of that region"""
#     if result == None:
#         result = {}

#     for d in dicts:
#         children = d.get("children")
#         # result[d['id']] = d['name']
#         result[d["id"]] = d

#         if children:
#             counter += 1
#             iterate_dicts(children, result=result, counter=counter)
#     return result


# def get_attributes_from_ids(ontology_by_ids, list_of_ids, get_keys):
#     assert isinstance(get_keys, list)
#     return [
#         get_attributes_for_list_of_ids(
#             ontology_by_ids, list_of_ids, get_key, warn=False
#         )
#         for get_key in get_keys
#     ]


# def get_attributes_for_list_of_ids(ontology_by_ids, list_of_ids, get_key, warn=False):
#     """though some atlas regions do not appear in ontology they seem to be insignificant"""
#     extracted_attributes = []
#     atlas_region_lookup_error_ids = []
#     for x in list_of_ids:
#         try:
#             val = extract_ont_label_key(ontology_by_ids, x, get_key)
#         except KeyError:
#             atlas_region_lookup_error_ids.append(x)
#             val = "notFound"
#         extracted_attributes.append(val)
#     lookup_error_ids_unique = set(atlas_region_lookup_error_ids)
#     if atlas_region_lookup_error_ids and warn:
#         print(
#             f"WARN n unique atlas ids not found: {len(lookup_error_ids_unique)} ({lookup_error_ids_unique}) total: ({len(atlas_region_lookup_error_ids)})"
#         )
#     return extracted_attributes


# def extract_ont_label_key(ontology_by_ids, px_label, get_key="acronym"):
#     """look up atlas id in ontology, returns acronym but can also return any other key such as name"""
#     values = ontology_by_ids[px_label]
#     if "data" in values and get_key != "children":  # e.g. for v3p1 ontology
#         values = values["data"]
#     return values[get_key]


# def find_empty_children(d):
#     empty_children = []

#     if isinstance(d, dict):
#         if "children" in d and not d["children"]:
#             empty_children.append(d)

#         for value in d.values():
#             empty_children.extend(find_empty_children(value))
#     elif isinstance(d, list):
#         for item in d:
#             empty_children.extend(find_empty_children(item))

#     return empty_children


# def extract_ids_breadth_first(data):
#     ids = []
#     queue = deque([data])

#     while queue:
#         current = queue.popleft()

#         if isinstance(current, dict):
#             if "children" in current:
#                 for child in current["children"]:
#                     ids.append(child["id"])
#                     queue.append(child)
#         elif isinstance(current, list):
#             for item in current:
#                 queue.append(item)

#     return ids


# def get_children(ont_ids, id):
#     return [d["id"] for d in ont_ids[id]["children"]]


# def gather_ids_by_st_level(d, as_dict=False):
#     """returns list of tuples (stl, id) or dict where keys are stl
#     contains duplicates !!!
#     need to filter these out I belive, like so:
#     set_st_order = []
#     for el in st_order_ids:
#         if el not in set_st_order:
#             set_st_order.append(el)"""

#     ids_and_levels = []

#     if isinstance(d, dict):
#         if "id" in d and "st_level" in d:
#             ids_and_levels.append((d["st_level"], d["id"]))

#         for value in d.values():
#             ids_and_levels.extend(gather_ids_by_st_level(value))
#     elif isinstance(d, list):
#         for item in d:
#             ids_and_levels.extend(gather_ids_by_st_level(item))
#     if as_dict:
#         return list_of_tups_to_dict(ids_and_levels)
#     else:
#         return ids_and_levels


# def list_of_tups_to_dict(lot):
#     ch_st_levels = {}
#     for tup in lot:
#         stl, id = tup
#         if stl not in ch_st_levels:
#             ch_st_levels[stl] = []
#         ch_st_levels[stl].append(id)
#     return ch_st_levels


# def parent_ontology_at_st_level(ont_ids, st_level_parents, start_index=8):
#     # slice the ontology so the parent level is defined by a specific st_level
#     # st_level_parents should be an st_level (int)
#     st_order_dict = gather_ids_by_st_level(
#         ont_ids[start_index], as_dict=True
#     )  # restrict to basic regions

#     st_parents = []
#     for reg_id in st_order_dict[st_level_parents]:
#         st_parents.append(ont_ids[reg_id])

#     # import json
#     # with open('st_8_parent_ont_ids.json', 'w') as f:
#     #     json.dump(st_8_parents, f)

#     return st_parents


def get_all_parents(ont, reg_id, parent_ids=None, max_region_id=997):
    if parent_ids is None:
        parent_ids = []
    if (reg_id == max_region_id) or (reg_id is None):
        return parent_ids
    parent_id = ont.ont_ids[reg_id].get("parent_structure_id", None)
    parent_ids.append(parent_id)
    return get_all_parents(ont, parent_id, parent_ids=parent_ids)


# def get_all_children(nested_list):
#     # returns a list of all children ids, where nested list is children key of ont for a specific region_id
#     ids = []

#     def helper(nested_item):
#         if isinstance(nested_item, list):
#             for item in nested_item:
#                 helper(item)
#         elif isinstance(nested_item, dict):
#             if "id" in nested_item:
#                 ids.append(nested_item["id"])
#             if "children" in nested_item:
#                 helper(nested_item["children"])

#     assert isinstance(nested_list, list), "please pass a list"
#     helper(nested_list)
#     return ids


# def map_children2parent(ont_ids, ont_slice):
#     child_to_parent_mapping = {}
#     for d in ont_slice:
#         child_ids = get_all_children(d["children"])
#         child_names = get_attributes_for_list_of_ids(ont_ids, child_ids, "name")
#         for chn in child_names:
#             child_to_parent_mapping[chn] = d["name"]
#     return child_to_parent_mapping


# def filter_regions_by_st_level(ont_ids, ont_slice, max_st_lvl=None):
#     # get regions at a specific st_level
#     max_st_lvl = (
#         99 if max_st_lvl is None else max_st_lvl
#     )  # optionally cap st_lvl at a certain value
#     # st_order_ids = [el[1] for el in gather_ids_by_st_level(ont_slice, as_dict=False)] # replaced so cap could be implemented
#     set_st_order = []  # need to convert to a set because of redundancies in ont_ids during iteration (each region appears by itself and under parent)
#     for st_lvl, reg_id in gather_ids_by_st_level(ont_slice, as_dict=False):
#         if (reg_id not in set_st_order) and (st_lvl <= max_st_lvl):
#             set_st_order.append(reg_id)
#     st_order_names = get_attributes_for_list_of_ids(ont_ids, set_st_order, "name")
#     return st_order_names


# def get_st_parents(ont_ids, child_regions, lvl):
#     # get parent regions at a specific st_level
#     ont_slice_colors = parent_ontology_at_st_level(ont_ids, lvl)
#     st_order_dict_colors = gather_ids_by_st_level(ont_slice_colors, as_dict=True)
#     child2parent_mapping_colors = {
#         k: v
#         for k, v in map_children2parent(ont_ids, ont_slice_colors).items()
#         if k in list(child_regions)
#     }
#     return child2parent_mapping_colors


    