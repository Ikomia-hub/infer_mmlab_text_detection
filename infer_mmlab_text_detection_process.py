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

import cv2
import torch
from ikomia import utils, core, dataprocess
import numpy as np
import copy
import os
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
from mmocr.utils import register_all_modules

# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferMmlabTextDetectionParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.update = False
        self.weights = ""
        self.model_cfg = "dbnet/dbnet_resnet50_1200e_icdar2015.py"
        self.deploy_cfg = "text-detection/text-detection_onnxruntime_dynamic.py"

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.update = utils.strtobool(param_map["update"])
        self.weights = param_map["weights"]
        self.model_cfg = param_map["model_cfg"]
        self.deploy_cfg = param_map["deploy_cfg"]

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        param_map["update"] = str(self.update)
        param_map["weights"] = str(self.weights)
        param_map["model_cfg"] = self.model_cfg
        param_map["deploy_cfg"] = self.deploy_cfg
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
        self.device ='cpu'
        self.input_shape = None
        self.model_cfg_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                        "mmocr", "configs","textdet")
        self.deploy_cfg_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                        "mmdeploy", "configs", "mmocr")

        # Create parameters class
        if param is None:
            self.setParam(InferMmlabTextDetectionParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self):
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
            # Get config files and model path
            model_cfg = os.path.join(self.model_cfg_path, param.model_cfg)
            deploy_cfg = os.path.join(self.deploy_cfg_path, param.deploy_cfg)
            backend_files = [param.weights]

            # read deploy_cfg and model_cfg
            deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)
            # build task
            register_all_modules()
            self.task_processor = build_task_processor(model_cfg, deploy_cfg, self.device)
            # process input image and backend model
            self.model = self.task_processor.build_backend_model(backend_files)
            print("Model loaded!")
            # process input image
            self.input_shape = get_input_shape(deploy_cfg)
            param.update = False

        if self.model is not None:
            if img is not None:
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                forwarded_output.setImage(img)
                # process input image
                model_inputs, _ = self.task_processor.create_input(
                                                        #img,
                                                        imgs= input.sourceFilePath,
                                                        input_shape = self.input_shape)
                self.infer(img, graphics_output, numeric_output, model_inputs)
            else:
                print("No input image")
        else:
            print("No model loaded")

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()

    def infer(self, img, graphics_output, numeric_output, model_inputs):
        color = [255, 0, 0]
        detected_names = []
        detected_conf = []
        h, w = np.shape(img)[:2]
        out = self.model.test_step(model_inputs)

        # Transform model output in an Ikomia format to be displayed
        polygons = out[0].pred_instances.polygons
        confidences = out[0].pred_instances.scores.numpy()
        for polygon, conf in zip(polygons, confidences):
            pts = np.array(polygon, dtype=float)
            pts = [core.CPointF(self.clamp(x, 0, w),
                                self.clamp(y, 0, h)) for x, y in zip(pts[0::2],
                                                                     pts[1::2])]
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
        self.info.shortDescription = "Inference for MMOCR from MMLAB text detection models in "\
                                    ".onnx format."
        self.info.description = "Models should be in .onnx format. Make sure you give to the " \
                                "plugin the  corresponding model config file (.py) and deploy " \
                                "config file (.py). If a costum (non-listed) config file is used," \
                                " it should saved in the appropriate config folder of the plugin."
                                
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Text"
        self.info.version = "1.0.1"
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
        self.info.keywords = "mmlab, mmocr, text, detection, onnx, dbnet, mask-rcnn, textsnake, pan-net, drrg, " \
                             "pse-net"

    def create(self, param=None):
        # Create process object
        return InferMmlabTextDetection(self.info.name, param)