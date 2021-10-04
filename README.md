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