import time

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


def run():
    torch.multiprocessing.freeze_support()
    #add "test"
    for d in ["train", "val", "test"]:
        DatasetCatalog.register("historical_" + d, lambda d=d: get_annots("../dataset", d))
        MetadataCatalog.get("historical_" + d).set(
            thing_classes=["soldier", "corpse", "person with KZ uniform", "crowd", "civilian"])
    historical_metadata = MetadataCatalog.get("historical_train")
    from detectron2.engine import DefaultTrainer

    cfg = get_cfg()
    # print('{}'.format(cfg.OUTPUT_DIR))
    # cfg.MODEL.DEVICE = "cpu"

    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("historical_train",)
    cfg.DATASETS.TEST = ("historical_test",) #("historical_val",)
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

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold

    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("historical_test", cfg, False, output_dir="./output/") #change to "historical_test"
    # DefaultTrainer.test(cfg, model=trainer.model, evaluators=evaluator)
    val_loader = build_detection_test_loader(cfg, "historical_test") #change to "historical_test"
    inference_on_dataset(trainer.model, val_loader, evaluator)

    from detectron2.utils.visualizer import ColorMode
    dataset_dicts = get_annots("../dataset", "test") #change to "test"
    total_time = 0
    '''for d in dataset_dicts:
        im = cv2.imread(d["file_name"])
        start = time.time()
        outputs = predictor(
            im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        total_time += time.time() - start
    print("Total inference time: {}".format(total_time))'''
    for d in reversed(dataset_dicts):
        im = cv2.imread(d["file_name"])
        outputs = predictor(
            im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                       metadata=historical_metadata,
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW
                       # remove the colors of unsegmented pixels. This option is only available for segmentation models
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.show()

    '''from detectron2.utils.visualizer import ColorMode
    dataset_dicts = get_annots("../dataset", "val") #change to "test"
    for d in random.sample(dataset_dicts, 30):
        im = cv2.imread(d["file_name"])
        outputs = predictor(
            im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                       metadata=historical_metadata,
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW
                       # remove the colors of unsegmented pixels. This option is only available for segmentation models
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.show()'''

if __name__ == '__main__':
    run()
'''
    for idx, v in enumerate(imgs_anns["assets"].values()):
        record = {}
        if mode == "train":
            p = os.path.join(dir, mode)
            filename = os.path.join(p, v["asset"]["name"])
        elif mode == "val":
            p = os.path.join(dir, mode)
            filename = os.path.join(p, v["asset"]["name"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        
        annos = v[]

    res_file = {
        "categories": categories,
        "images": [],
        "annotations": []
    }

    count = 0
    for image_data in imgs_anns["assets"].values():
        annotations = []
        if len(image_data["regions"]) == 0:
            continue

        num_regions = len(image_data["regions"])
        for i in range(num_regions):
            annot_elem = {
                "category_id": cat_dict[image_data["regions"][i]["tags"][0]],
                "bbox": [image_data["regions"][i]["boundingBox"]["left"],
                         image_data["regions"][i]["boundingBox"]["top"],
                         image_data["regions"][i]["boundingBox"]["left"] + image_data["regions"][i]["boundingBox"][
                             "width"],
                         image_data["regions"][i]["boundingBox"]["top"] + image_data["regions"][i]["boundingBox"][
                             "height"]],
                "iscrowd": 0,
                "bbox_mode": boxes.BoxMode.XYXY_ABS
            }
            annotations.append(annot_elem)
        dataset_dicts[count]["annotations"] = annotations
        count += 1

register_coco_instances("historical_train", {}, "../dataset/SUS_train.json", "../dataset/train")
register_coco_instances("historical_val", {}, "../dataset/SUS_val.json", "../dataset/val")

#dataset_dicts = detectron2.data.datasets.load_coco_json("../dataset/SUS_train.json", "../dataset/train", dataset_name="historical_train")

from detectron2.data import MetadataCatalog
MetadataCatalog.get("historical_train").thing_classes = ["soldier", "corpse", "person with KZ uniform", "crowd", "civilian"]
dataset_dicts = DatasetCatalog.get("historical_train")

#print('{}'.format(dataset_dicts))'''
'''
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

json_file = 'C:\\Users\\chris\\Desktop\\SUS-train-export.json'
with open(json_file) as f:
    imgs_anns = json.load(f)

res_file = {
    "categories": categories,
    "images": [],
    "annotations": []
}

count = 0
for image_data in imgs_anns["assets"].values():
    annotations = []
    if len(image_data["regions"]) == 0:
        continue

    num_regions = len(image_data["regions"])
    for i in range(num_regions):
        annot_elem = {
            "category_id": cat_dict[image_data["regions"][i]["tags"][0]],
            "bbox": [image_data["regions"][i]["boundingBox"]["left"], image_data["regions"][i]["boundingBox"]["top"],image_data["regions"][i]["boundingBox"]["left"] + image_data["regions"][i]["boundingBox"]["width"], image_data["regions"][i]["boundingBox"]["top"] + image_data["regions"][i]["boundingBox"]["height"]],
            "iscrowd": 0,
            "bbox_mode": boxes.BoxMode.XYXY_ABS
        }
        annotations.append(annot_elem)
    dataset_dicts[count]["annotations"] = annotations
    count += 1

sampled = random.sample(dataset_dicts, 3)

#print('{}'.format(sampled))

for d in sampled:
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:,:,::-1], scale=.5)
    vis = visualizer.draw_dataset_dict(d)
    plt.imshow(vis.get_image()[:,:,::-1])
    plt.show()
'''
'''
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("C:\\Users\\chris\\Desktop\\coco_eval", exist_ok=True)
        output_folder = "C:\\Users\\chris\\Desktop\\coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)

from detectron2.config import get_cfg
#from detectron2.evaluation.coco_evaluation import COCOEvaluator
import os

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("historical_train",)
cfg.DATASETS.TEST = ("historical_val",)

cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001


#cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 30000 #adjust up if val mAP is still rising, adjust down if overfit
#cfg.SOLVER.STEPS = (1000, 1500)
#cfg.SOLVER.GAMMA = 0.05


#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5 #your number of classes + 1

#cfg.TEST.EVAL_PERIOD = 500


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()'''