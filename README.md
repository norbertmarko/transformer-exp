## Overview

    Requirements:
        - Python 3.9
        - CUDA 11.1
        - PyTorch 1.9 (+ TorchVision 0.10.1)


## Installation

    # Add dead-snakes repository
    sudo add-apt-repository ppa:deadsnakes/ppa

    # Install Python 3.9, development package and the specific venv:
    sudo apt install python3.9
    sudo apt install python3.9-dev
    sudo apt install python3.9-venv

    mkvirtualenv -p python3.9 maskformer

    pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f \ 
        https://download.pytorch.org/whl/torch_stable.html

    python -m pip install detectron2 -f \
        https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

    pip install opencv-python

    pip install -r requirements.txt

## Commands

    # Press ENTER between predictions!

    cd demo/
    python demo.py --config-file  ../configs/mapillary-vistas-65/maskformer_R50_bs16_300k.yaml \
        --input ../INPUTS/frame1.png ../INPUTS/frame2.png \
        --opts MODEL.WEIGHTS ../PRETRAINED/mapillary/model_final_f3fc73.pkl

## Custom Dataset training

https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html

Off-road robotics: https://unmannedlab.github.io/research/RELLIS-3D

## Custom Dataset notes


1. Register your dataset (i.e., tell detectron2 how to obtain your dataset).

2. Optionally, register metadata for your dataset.

| Task | Fields |
|------|--------|
|Common| file_name, height, width, image_id |
|Semantic segmentation | sem_seg_file_name |

Field Descriptions:

- file_name: the full path to the image file.

- height, width: integer. The shape of the image.

- image_id (str or int): a unique id that identifies this image. Required by many evaluators to identify the images, but a dataset may use it for different purposes.

- sem_seg_file_name (str): The full path to the semantic segmentation ground truth file. It should be a grayscale image whose pixel values are integer labels.

### MetaData

If you register a new dataset through DatasetCatalog.register, you may also want to add its corresponding metadata:

Metadata keys that are used by builtin features in detectron2:

thing_classes (list[str]): Used by all instance detection/segmentation tasks. A list of names for each instance/thing category. If you load a COCO format dataset, it will be automatically set by the function load_coco_json.

thing_colors (list[tuple(r, g, b)]): Pre-defined color (in [0, 255]) for each thing category. Used for visualization. If not given, random colors will be used.

ignore_label (int): Used by semantic and panoptic segmentation tasks. Pixels in ground-truth annotations with this category label should be ignored in evaluation. Typically these are “unlabeled” pixels.