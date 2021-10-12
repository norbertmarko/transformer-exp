from pathlib import Path
import cv2

from typing import List, Dict
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog


def my_dataset_function() -> List[Dict]:
    """
    The function can do arbitrary things 
    and should return the data in list[dict].
    """
    # object to be returned
    dataset_dicts = []

    root_path =  Path(r"/mnt/data_disk/datasets/rellis-3d/Rellis-3D/")
    train_list_path = '/mnt/data_disk/datasets/rellis-3d/train.lst'
    train_list = open(train_list_path)

    for idx, line in enumerate(train_list):
        record = {}
        
        img_path = str( root_path / line.strip().split()[0] )
        label_path = str( root_path / line.strip().split()[1] )
        
        height, width = cv2.imread(img_path).shape[:2]

        record["file_name"] = img_path
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        record["sem_seg_file_name"] = label_path

        dataset_dicts.append(record)

        print(idx)

    return dataset_dicts


# register the dataset
DatasetCatalog.register("Rellis-3D", my_dataset_function)


# add metadata
MetadataCatalog.get("Rellis-3D").thing_classes = [
    "void", "grass", "tree", "pole", "water", "sky",
    "vehicle", "object", "asphalt", "building", "log",
    "person", "fence", "bush", "concrete", "barrier",
    "puddle", "mud", "rubble"
]

MetadataCatalog.get("Rellis-3D").ignore_label = 0 

# later, to access the data:
data: List[Dict] = DatasetCatalog.get("Rellis-3D")

print(data[0])