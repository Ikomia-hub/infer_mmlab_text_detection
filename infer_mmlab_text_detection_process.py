# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import os.path

import cv2
import torch.cuda
from ikomia import utils, core, dataprocess
from mmocr.apis.inferencers import TextDetInferencer
import numpy as np
import copy
import os
from mmocr.utils import register_all_modules
import yaml


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferMmlabTextDetectionParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.update = False
        self.config_file = ""
        self.model_name = "dbnet"
        self.model_weight_file = ""
        self.cfg = "dbnet_resnet18_fpnc_1200e_icdar2015"
        self.model_url = "https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015/" \
                         "dbnet_resnet18_fpnc_1200e_icdar2015_20220825_221614-7c0e94f2.pth"
        self.use_custom_model = False

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.config_file = param_map["config_file"]
        self.update = utils.strtobool(param_map["update"])
        self.model_name = param_map["model_name"]
        self.cfg = param_map["cfg"]
        self.model_url = param_map["model_url"]
        self.use_custom_model = utils.strtobool(param_map["use_custom_model"])
        self.model_weight_file = param_map["model_weight_file"]

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        param_map["config_file"] = self.config_file
        param_map["update"] = str(self.update)
        param_map["model_name"] = self.model_name
        param_map["cfg"] = self.cfg
        param_map["model_url"] = self.model_url
        param_map["use_custom_model"] = str(self.use_custom_model)
        param_map["model_weight_file"] = self.model_weight_file
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferMmlabTextDetection(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the process here
        # Add graphics output
        self.add_output(dataprocess.CTextIO())

        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create parameters class
        if param is None:
            self.set_param_object(InferMmlabTextDetectionParam())
        else:
            self.set_param_object(copy.deepcopy(param))

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()
        param = self.get_param_object()
        # Get input :
        input = self.get_input(0)
        img = input.get_image()

        # Get output :
        text_output = self.get_output(1)

        # clear output before each run as temporary fix
        text_output.clear_data()

        forwarded_output = self.get_output(0)

        # Load models into memory if needed
        if self.model is None or param.update:
            if param.model_weight_file != "":
                param.use_custom_model = True
    
            if not param.use_custom_model:
                yaml_file = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "textdet"), param.model_name, "metafile.yml")

                if os.path.isfile(yaml_file):
                    with open(yaml_file, "r") as f:
                        models_list = yaml.load(f, Loader=yaml.FullLoader)['Models']

                    available_cfg_ckpt = {model_dict["Name"]: {'cfg': model_dict["Config"],
                                                                    'ckpt': model_dict["Weights"]}
                                               for model_dict in models_list}
                    if param.cfg in available_cfg_ckpt:
                        cfg = available_cfg_ckpt[param.cfg]['cfg']
                        ckpt = available_cfg_ckpt[param.cfg]['ckpt']
                        cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg)
                    else:
                        raise Exception(f"{param.cfg} dos not exist for {param.model_name}. Available configs for are {', '.join(list(available_cfg_ckpt.keys()))}")
                else:
                    raise Exception(f"Model name {param.model_name} does not exist.")
            else:
                if os.path.isfile(param.cfg):
                    cfg = param.cfg
                else:
                    cfg = param.config_file
                ckpt = param.model_weight_file
            register_all_modules()
            self.model = TextDetInferencer(cfg, ckpt, device=self.device)

            param.update = False
            print("Model loaded!")

        if self.model is not None:
            if img is not None:
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                forwarded_output.set_image(img)
                self.infer(img, text_output)
            else:
                print("No input image")
        else:
            print("No model loaded")

        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()

    def infer(self, img, text_output):
        color = [255, 0, 0]
        h, w = np.shape(img)[:2]
        out = self.model(img)['predictions'][0]

        # Transform model output in an Ikomia format to be displayed
        for i, (polygon, conf) in enumerate(zip(out['polygons'], out['scores'])):
            pts = np.array(polygon, dtype=float)
            pts = [core.CPointF(self.clamp(x, 0, w), self.clamp(y, 0, h)) for x, y in zip(pts[0::2], pts[1::2])]

            text_output.add_text_field(id=i, label="", text="", confidence=float(conf), polygon=pts, color=color )

        text_output.finalize()

    @staticmethod
    def clamp(x, min_val, max_val):
        return min_val if x < min_val else max_val - 1 if x > max_val - 1 else x

    def stop(self):
        super().stop()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferMmlabTextDetectionFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_mmlab_text_detection"
        self.info.short_description = "Inference for MMOCR from MMLAB text detection models"
        self.info.description = "If custom training is disabled, models will come from MMLAB's model zoo." \
                                "Else, you can also choose to load a model you trained yourself with our plugin " \
                                "train_mmlab_text_detection. In this case make sure you give to the plugin" \
                                "a config file (.py) and a model file (.pth). Both of these files are produced " \
                                "by the train plugin."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Text"
        self.info.version = "1.1.2"
        self.info.icon_path = "icons/mmlab.png"
        self.info.authors = "Kuang, Zhanghui and Sun, Hongbin and Li, Zhizhong and Yue, Xiaoyu and Lin," \
                            " Tsui Hin and Chen, Jianyong and Wei, Huaqiang and Zhu, Yiqin and Gao, Tong and Zhang," \
                            " Wenwei and Chen, Kai and Zhang, Wayne and Lin, Dahua"
        self.info.article = "MMOCR:  A Comprehensive Toolbox for Text Detection, Recognition and Understanding"
        self.info.journal = "Arxiv"
        self.info.year = 2021
        self.info.license = "Apache-2.0 License"
        # URL of documentation
        self.info.documentation_link = "https://mmocr.readthedocs.io/en/latest/"
        # Code source repository
        self.info.repository = "https://github.com/open-mmlab/mmocr"
        # Keywords used for search
        self.info.keywords = "mmlab, mmocr, text, detection, pytorch, dbnet, mask-rcnn, textsnake, pan-net, drrg, " \
                             "pse-net"

    def create(self, param=None):
        # Create process object
        return InferMmlabTextDetection(self.info.name, param)
