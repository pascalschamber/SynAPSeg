import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # add parent dir to sys.path
from SynAPSeg.IO.BaseConfig import BaseConfig
from SynAPSeg.config import constants
from SynAPSeg.utils import utils_image_processing as uip
from SynAPSeg.utils import utils_plotting as up
from SynAPSeg.utils import utils_general as ug
from SynAPSeg.Segmentation.Processing import init_ex_md
from SynAPSeg.IO.image_parser import ImageParser
from SynAPSeg.IO.env import verify_and_set_env_dirs

# test image parser 
if __name__ == "__main__":

    verify_and_set_env_dirs()
    #TODO: replace ex below with demo image path and values

    test_img_path = r"D:\OneDrive - Tufts\Classes\Rotation\BygraveLab\SynAPSeg\tests\data\test_img_CZYX_2-13-512-512.tiff"
    expected_shape_final = (1, 1, 2, 13, 512, 512)

    SEG_CONFIG = BaseConfig(
            'test', 
            params=dict(
                image_path=test_img_path, input_image_format='CZYX',
                LOG_RUN=False, WRITE_OUTPUT=False,

            ), 
            default_parameters_path=constants.SEG_DEFAULT_PARAMETERS_PATH)
    SEG_CONFIG.resolve_unspecified_default_parameters(SEG_CONFIG.default_parameters_path, SEG_CONFIG.params)

    ex_md = init_ex_md(0, test_img_path, SEG_CONFIG)
    image_parser = ImageParser.create_parser(test_img_path, ex_md)

    image_obj, arr, ex_md = image_parser.run()
    # arr, arr_mip = format_prediction_input(image_parser, img_obj, arr, ex_md, SEG_CONFIG)

    arr, arr_mip = image_parser.try_format_prediction_input(image_obj, arr, ex_md, SEG_CONFIG)
    assert arr.shape == expected_shape_final, (f"{arr.shape} != {expected_shape_final}")
