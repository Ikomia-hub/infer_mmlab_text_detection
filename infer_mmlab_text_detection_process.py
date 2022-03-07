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
from ikomia import core, dataprocess
from mmocr.apis.inference import *
import numpy as np
import copy
import distutils
from mmcv import Config
from infer_mmlab_text_detection.utils import textdet_models


# Your imports below


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferMmlabTextDetectionParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.update = False
        self.model_name = "DB_r50"
        self.cfg = ""
        self.weights = ""
        self.custom_training = False

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.update = distutils.util.strtobool(param_map["update"])
        self.model_name = param_map["model_name"]
        self.cfg = param_map["cfg"]
        self.weights = param_map["weights"]
        self.custom_training = distutils.util.strtobool(param_map["custom_training"])

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        param_map["update"] = str(self.update)
        param_map["model_name"] = self.model_name
        param_map["cfg"] = self.cfg
        param_map["weights"] = self.weights
        param_map["custom_training"] = str(self.custom_training)
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
        self.addOutput(dataprocess.CGraphicsOutput())
        # Add numeric output
        self.addOutput(dataprocess.CNumericIO())

        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create parameters class
        if param is None:
            self.setParam(InferMmlabTextDetectionParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()
        param = self.getParam()
        # Get input :
        input = self.getInput(0)
        img = input.getImage()

        # Get output :
        graphics_output = self.getOutput(1)

        # Init numeric and graphics outputs
        numeric_output = self.getOutput(2)
        graphics_output.setNewLayer("mmlab_text_detection")
        graphics_output.setImageIndex(0)
        numeric_output.clearData()
        numeric_output.setOutputType(dataprocess.NumericOutputType.TABLE)
        forwarded_output = self.getOutput(0)

        # Load models into memory if needed
        if self.model is None or param.update:
            device = torch.device(self.device)
            if not (param.custom_training):
                cfg = Config.fromfile(os.path.join(os.path.dirname(__file__), "configs/textdet",
                                                   textdet_models[param.model_name]["config"]))
                ckpt = os.path.join('https://download.openmmlab.com/mmocr/textdet/',
                                    textdet_models[param.model_name]["ckpt"])
            else:
                cfg = Config.fromfile(param.cfg)
                ckpt = param.weights if param.weights != "" and param.custom_training else None

            cfg.test_pipeline[0]['type'] = 'LoadImageFromNdarray'
            cfg.test_pipeline[1].img_scale = (2000, 1800)
            cfg.data.test.pipeline = cfg.test_pipeline
            self.model = init_detector(cfg, ckpt, device=device)

            param.update = False
            print("Model loaded!")

        if self.model is not None:
            if img is not None:
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                forwarded_output.setImage(img)
                self.infere(img, graphics_output, numeric_output)
            else:
                print("No input image")
        else:
            print("No model loaded")

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()

    def infere(self, img, graphics_output, numeric_output):
        color = [255, 0, 0]
        detected_names = []
        detected_conf = []
        h, w = np.shape(img)[:2]
        out = model_inference(self.model,
                              img,
                              ann=None,
                              batch_mode=False,
                              return_data=True)
        boundary_result = out[0]['boundary_result']

        # Transform model output in an Ikomia format to be displayed
        for polygone_conf in boundary_result:
            pts = np.array(polygone_conf[:-1], dtype=float)
            pts = [core.CPointF(self.clamp(x, 0, w), self.clamp(y, 0, h)) for x, y in zip(pts[0::2], pts[1::2])]
            conf = polygone_conf[-1]
            prop_poly = core.GraphicsPolygonProperty()
            prop_poly.pen_color = color
            graphics_box = graphics_output.addPolygon(pts, prop_poly)
            graphics_box.setCategory('text')
            # Label
            name = 'text'
            detected_names.append(name)
            detected_conf.append(float(conf))

        numeric_output.addValueList(detected_conf, "Confidence", detected_names)

    def clamp(self, x, min, max):
        return min if x < min else max - 1 if x > max - 1 else x

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
        self.info.shortDescription = "Inference for MMOCR from MMLAB text detection models"
        self.info.description = "If custom training is disabled, models will come from MMLAB's model zoo." \
                                "Else, you can also choose to load a model you trained yourself with our plugin " \
                                "train_mmlab_text_detection. In this case make sure you give to the plugin" \
                                "a config file (.py) and a model file (.pth). Both of these files are produced " \
                                "by the train plugin."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Text"
        self.info.version = "1.0.0"
        self.info.iconPath = "icons/mmlab.png"
        self.info.authors = "Kuang, Zhanghui and Sun, Hongbin and Li, Zhizhong and Yue, Xiaoyu and Lin," \
                            " Tsui Hin and Chen, Jianyong and Wei, Huaqiang and Zhu, Yiqin and Gao, Tong and Zhang," \
                            " Wenwei and Chen, Kai and Zhang, Wayne and Lin, Dahua"
        self.info.article = "MMOCR:  A Comprehensive Toolbox for Text Detection, Recognition and Understanding"
        self.info.journal = "Arxiv"
        self.info.year = 2021
        self.info.license = "Apache-2.0 License"
        # URL of documentation
        self.info.documentationLink = "https://mmocr.readthedocs.io/en/latest/"
        # Code source repository
        self.info.repository = "https://github.com/open-mmlab/mmocr"
        # Keywords used for search
        self.info.keywords = "mmlab, mmocr, text, detection, pytorch, dbnet, mask-rcnn, textsnake, pan-net, drrg, " \
                             "pse-net"

    def create(self, param=None):
        # Create process object
        return InferMmlabTextDetection(self.info.name, param)
