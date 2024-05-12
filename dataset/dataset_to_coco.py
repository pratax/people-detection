import os
import json
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--file', type=str)
    parser.add_argument('--mode', type=str)
    return parser


def run(file, mode):
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

    json_file = file
    with open(json_file) as f:
        imgs_anns = json.load(f)

    save_json = os.path.join(os.getcwd(), f"SUS_{mode}.json")

    id = 0
    annot_count = 0
    res_file = {
        "categories": categories,
        "images": [],
        "annotations": []
    }

    for image_data in imgs_anns["assets"].values():
        img_elem = {
            "file_name": image_data["asset"]["name"],
            "height": image_data["asset"]["size"]["height"],
            "width": image_data["asset"]["size"]["width"],
            "id": id
        }

        res_file["images"].append(img_elem)
        num_regions = len(image_data["regions"])

        for i in range(num_regions):
            poly = [[int(image_data["regions"][i]["points"][0]["x"]), int(image_data["regions"][i]["points"][0]["y"])],
                    [int(image_data["regions"][i]["points"][1]["x"]), int(image_data["regions"][i]["points"][1]["y"])],
                    [int(image_data["regions"][i]["points"][2]["x"]), int(image_data["regions"][i]["points"][2]["y"])],
                    [int(image_data["regions"][i]["points"][3]["x"]), int(image_data["regions"][i]["points"][3]["y"])]
                    ]

            annot_elem = {
                "id": annot_count,
                "image_id": id,
                "category_id": cat_dict[image_data["regions"][i]["tags"][0]],
                "bbox": [image_data["regions"][i]["boundingBox"]["left"], image_data["regions"][i]["boundingBox"]["top"], image_data["regions"][i]["boundingBox"]["width"], image_data["regions"][i]["boundingBox"]["height"]],
                "area": image_data["asset"]["size"]["height"]*image_data["asset"]["size"]["width"],
                "segmentation": list([poly]),
                "iscrowd": 0
            }

            res_file["annotations"].append(annot_elem)
            annot_count += 1
        id += 1

    print('total images: ', len(res_file["images"]), ' total annotations: ', len(res_file["annotations"]))

    with open(save_json, "w") as f:
        json_str = json.dumps(res_file)
        f.write(json_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RCNN evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    run(args.file, args.mode)
