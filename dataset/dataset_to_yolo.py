import json
import os
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--mode', type=str)
    return parser


def run(phase):
    cat_dict = {
        "soldier": 0,
        "corpse": 1,
        "person with KZ uniform": 2,
        "crowd": 3,
        "civilian": 4
    }

    json_file = os.path.join(os.getcwd(), f'SUS-{phase}-export.json')
    with open(json_file) as f:
        imgs_anns = json.load(f)

    save_json = os.path.join(os.getcwd(), f"yolo\\labels\\{phase}")
    total_images = 0
    total_annotations = 0
    for image_data in imgs_anns["assets"].values():
        height, width = image_data["asset"]["size"]["height"], image_data["asset"]["size"]["width"]
        name = image_data["asset"]["name"].split(".")[0]
        f = open(os.path.join(save_json, f'{name}.txt'), "w+")
        num_regions = len(image_data["regions"])
        total_annotations += num_regions
        for i in range(num_regions):
            category = cat_dict[image_data["regions"][i]["tags"][0]]
            x_center = (image_data["regions"][i]["boundingBox"]["left"] + (image_data["regions"][i]["boundingBox"]["width"] / 2)) / width
            y_center = (image_data["regions"][i]["boundingBox"]["top"] + (image_data["regions"][i]["boundingBox"]["height"] / 2)) / height
            w = image_data["regions"][i]["boundingBox"]["width"] / width
            h = image_data["regions"][i]["boundingBox"]["height"] / height
            f.write(f"{category} {x_center} {y_center} {w} {h} \n")
        total_images += 1
    print('total images: ', total_images, ' total annotations: ', total_annotations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RCNN evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    run(args.mode)
