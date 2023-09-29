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

![Result example](https://raw.githubusercontent.com/Ikomia-hub/infer_mmlab_text_detection/main/icons/results.jpg)

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

# Get results
original_image_output = algo.get_output(0)
text_detection_output = algo.get_output(1)

# Display results
display(original_image_output.get_image_with_graphics(text_detection_output))
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

# Get results
original_image_output = algo.get_output(0)
text_detection_output = algo.get_output(1)

# Display results
display(original_image_output.get_image_with_graphics(text_detection_output))
```

To know what are all the available pairs (**model_name**, **cfg**), run this code snippet.

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_mmlab_text_detection", auto_connect=True)

# Get possible parameters
possible_parameters = algo.get_model_zoo()

# Print them
print(possible_parameters)

# You can use one of them to choose your pretrain, here the first in the list
algo.set_parameters(possible_parameters[0])

# Then run on your image...
```

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
