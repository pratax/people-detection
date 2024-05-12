import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import argparse

import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../people_detection/detectron2')
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances

import detectron2.structures.boxes as boxes

from detectron2.evaluation import COCOEvaluator, inference_on_dataset

import json
import os
import detectron2.structures.boxes as boxes
import torch

def get_annots(dir, mode):
    if mode == "train":
        json_file = os.path.join(dir, 'SUS-train-export.json')
    elif mode == "val":
        json_file = os.path.join(dir, 'SUS-val-export.json')
    elif mode == "test":
        json_file = os.path.join(dir, 'SUS-test-export.json')

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
        elif mode == "test":
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

def get_annots_inf(files):
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
    #print('{}'.format(list))
    for file in files:
        record = {}
        filename = file
        height, width = cv2.imread(filename).shape[:2]
        record["file_name"] = filename
        record["image_id"] = count
        record["height"] = height
        record["width"] = width
        objs = []
        record["annotations"] = objs
        dataset_dicts.append(record)
        count += 1
    return dataset_dicts


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--file', type=str)

    return parser


def run(file):
    torch.multiprocessing.freeze_support()
    #add "test"
    files = []
    for filename in os.listdir(file):
        f = os.path.join(file, filename)
        files.append(f)
    for d in ["test"]:
        DatasetCatalog.register("historical_" + d, lambda d=d: get_annots_inf(files))
        MetadataCatalog.get("historical_" + d).set(
            thing_classes=["soldier", "corpse", "person with KZ uniform", "crowd", "civilian"])
    historical_metadata = MetadataCatalog.get("historical_test")
    from detectron2.engine import DefaultTrainer

    cfg = get_cfg()
    # print('{}'.format(cfg.OUTPUT_DIR))
    # cfg.MODEL.DEVICE = "cpu"

    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    #cfg.DATASETS.TRAIN = ("historical_train",)
    #cfg.DATASETS.TEST = ("historical_test",) #("historical_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 1000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.TEST.EVAL_PERIOD = 500

    cfg.MODEL.WEIGHTS = "../people_detection/detectron2/output/model_final.pth"  # path to the model we just trained

    #trainer = DefaultTrainer(cfg)
    #trainer.resume_or_load(resume=False)

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold

    predictor = DefaultPredictor(cfg)

    from detectron2.utils.visualizer import ColorMode
    #dataset_dicts = get_annots("../dataset", "test") #change to "test"
    dataset_dicts = get_annots_inf(files)
    import time

    for d in dataset_dicts:
        im = cv2.imread(d["file_name"])
        start = time.time()
        outputs = predictor(
            im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        end = time.time()
        v = Visualizer(im[:, :, ::-1],
                       metadata=historical_metadata,
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW
                       # remove the colors of unsegmented pixels. This option is only available for segmentation models
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.show()
    print('total inference time {}'.format(end-start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RCNN evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    run(args.file)
