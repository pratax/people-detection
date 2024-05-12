import time
import sys
sys.path.insert(0, '../people_detection/detectron2')
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances

from detectron2.evaluation import COCOEvaluator, inference_on_dataset

import detectron2.structures.boxes as boxes

import json
import os
import detectron2.structures.boxes as boxes
import torch

def get_annots(dir, mode):
    if mode == "train":
        json_file = os.path.join(dir, 'SUS-train-export.json')
    elif mode == "val":
        json_file = json_file = os.path.join(dir, 'SUS-val-export.json')

    with open(json_file) as f:
        imgs_anns = json.load(f)

    categories = [
        {"supercategory": "none", "name": "soldier", "id": 0, "color": "#5db300"},
        {"supercategory": "none", "name": "corpse", "id": 1, "color": "#e81123"},
        {"supercategory": "none", "name": "person with KZ uniform", "id": 2, "color": "#6917aa"},
        {"supercategory": "none", "name": "crowd", "id": 3, "color": "#015cda"},
        {"supercategory": "none", "name": "civilian", "id": 4, "color": "#4894fe"},
    ]

    cat_dict = {
        "soldier": 0,
        "corpse": 1,
        "person with KZ uniform": 2,
        "crowd": 3,
        "civilian": 4
    }

    dataset_dicts = []
    count = 0
    list = imgs_anns["assets"].values()
    #print('{}'.format(list))
    for image_data in imgs_anns["assets"].values():
        record = {}
        if mode == "train":
            p = os.path.join(dir, mode)
            filename = os.path.join(p, image_data["asset"]["name"])
        elif mode == "val":
            p = os.path.join(dir, mode)
            filename = os.path.join(p, image_data["asset"]["name"])
        height, width = cv2.imread(filename).shape[:2]
        record["file_name"] = filename
        record["image_id"] = count
        record["height"] = height
        record["width"] = width

        annos = image_data["regions"]
        objs = []
        for anno in annos:
            poly = [[int(anno["points"][0]["x"]), int(anno["points"][0]["y"])],
                [int(anno["points"][1]["x"]), int(anno["points"][1]["y"])],
                [int(anno["points"][2]["x"]), int(anno["points"][2]["y"])],
                [int(anno["points"][3]["x"]), int(anno["points"][3]["y"])]
                ]
            obj = {
                "bbox": [anno["points"][0]["x"], anno["points"][0]["y"], anno["points"][2]["x"], anno["points"][2]["y"]],
                "bbox_mode": boxes.BoxMode.XYXY_ABS,
                #"segmentation": [poly],
                "category_id": cat_dict[anno["tags"][0]],
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
        count += 1
    return dataset_dicts


def run():
    torch.multiprocessing.freeze_support()
    for d in ["train", "val"]:
        DatasetCatalog.register("historical_" + d, lambda d=d: get_annots("../dataset", d))
        MetadataCatalog.get("historical_" + d).set(thing_classes=["soldier", "corpse", "person with KZ uniform", "crowd", "civilian"])
    historical_metadata = MetadataCatalog.get("historical_train")
    dataset_dicts = get_annots("../dataset", "train")

    from detectron2.engine import DefaultTrainer

    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("historical_train",)
    cfg.DATASETS.TEST = ("historical_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 30000
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.TEST.EVAL_PERIOD = 500

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    start = time.time()
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    print("Training time: {}".format(time.time()-start))

if __name__ == '__main__':
    import wandb
    wandb.init(project="faster rcnn", entity="pratax", sync_tensorboard=True)

    run()
