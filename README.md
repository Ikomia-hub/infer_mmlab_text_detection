<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/infer_mmlab_text_detection/main/icons/mmlab.png" alt="Algorithm icon">
  <h1 align="center">infer_mmlab_text_detection</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_mmlab_text_detection">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_mmlab_text_detection">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_mmlab_text_detection/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_mmlab_text_detection.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Run text detection models from MMLAB.

![Result example](https://raw.githubusercontent.com/Ikomia-hub/infer_mmlab_text_detection/feat/new_readme/icons/results.jpg)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_mmlab_text_detection", auto_connect=True)

# Run on your image
wf.run_on(url="https://discuss.poynt.net/uploads/default/original/2X/6/60c4199364474569561cba359d486e6c69ae8cba.jpeg")

# Get graphics
graphics = algo.get_output(1).get_graphics_io()

# Display results
display(algo.get_output(0).get_image_with_graphics(graphics))
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

- **model_name** (str, default="dbnet"): pre-trained model name. 
- **cfg** (str, default="dbnet_resnet18_fpnc_1200e_icdar2015.py"): config of the pretrained model.
- **cuda** (bool, default=True): CUDA acceleration if True, run on CPU otherwise.
- **config_file** (str, default=""): path to model config file (.py). Only for custom model.
- **model_weight_file** (str, default=""): path to model weights file (.pt). Only for custom model.

To run a specific pretrained model, fill **model_name** and **cfg**.
To run a custom model, for example trained with **train_mmlab_text_detection**, fill **config_file** and **model_weight_file**

***Note***: parameter key and value should be in **string format** when added to the dictionary.

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_mmlab_text_detection", auto_connect=True)

algo.set_parameters({
    "model_name": "dbnetpp",
    "cfg": "dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015",
    "cuda": "True"
})

# Run on your image  
wf.run_on(url="https://discuss.poynt.net/uploads/default/original/2X/6/60c4199364474569561cba359d486e6c69ae8cba.jpeg")

# Get graphics
graphics = algo.get_output(1).get_graphics_io()

# Display results
display(algo.get_output(0).get_image_with_graphics(graphics))
```

Find below the exhaustive list of available combinations of **model_name** and **cfg**:
- panet
  - panet_resnet18_fpem-ffm_600e_ctw1500
  - panet_resnet18_fpem-ffm_600e_icdar2015
- textsnake
  - textsnake_resnet50_fpn-unet_1200e_ctw1500
  - textsnake_resnet50-oclip_fpn-unet_1200e_ctw1500
- dbnetpp
  - dbnetpp_resnet50_fpnc_1200e_icdar2015
  - dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015
  - dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015
- maskrcnn
  - mask-rcnn_resnet50_fpn_160e_ctw1500
  - mask-rcnn_resnet50-oclip_fpn_160e_ctw1500
  - mask-rcnn_resnet50_fpn_160e_icdar2015
  - mask-rcnn_resnet50-oclip_fpn_160e_icdar2015
- drrg
  - drrg_resnet50_fpn-unet_1200e_ctw1500
- fcenet
  - fcenet_resnet50-dcnv2_fpn_1500e_ctw1500
  - fcenet_resnet50-oclip_fpn_1500e_ctw1500
  - fcenet_resnet50_fpn_1500e_icdar2015
  - fcenet_resnet50-oclip_fpn_1500e_icdar2015
  - fcenet_resnet50_fpn_1500e_totaltext
- dbnet
  - dbnet_resnet18_fpnc_1200e_icdar2015
  - dbnet_resnet50_fpnc_1200e_icdar2015
  - dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015
  - dbnet_resnet50-oclip_fpnc_1200e_icdar2015
  - dbnet_resnet18_fpnc_1200e_totaltext
- psenet
  - psenet_resnet50_fpnf_600e_ctw1500
  - psenet_resnet50-oclip_fpnf_600e_ctw1500
  - psenet_resnet50_fpnf_600e_icdar2015
  - psenet_resnet50-oclip_fpnf_600e_icdar2015

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_mmlab_text_detection", auto_connect=True)

# Run on your image  
wf.run_on(url="https://discuss.poynt.net/uploads/default/original/2X/6/60c4199364474569561cba359d486e6c69ae8cba.jpeg")

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```
