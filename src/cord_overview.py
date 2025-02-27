import json
import random

import click
import cv2
import numpy as np
from tqdm import tqdm

img_prefix = "/home/rmnv/dev/reciept-dissection/examples/train/image"
json_prefix = "/home/rmnv/dev/reciept-dissection/examples/train/json"

# "quad": {
#                         "x2": 619,
#                         "y3": 644,
#                         "x3": 620,
#                         "y4": 647,
#                         "x1": 543,
#                         "y1": 622,
#                         "x4": 544,
#                         "y2": 619
#                     },


def draw_bbos(img, b, color, text):
    pts = np.array([[int(b[f"x{i}"]), int(b[f"y{i}"])] for i in range(1, 5)])
    pts = pts.reshape(1, 1, 4, 2)
    cv2.drawContours(img, pts, -1, color, 3)
    cv2.putText(img, text, pts[0][0][0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, cv2.LINE_4)


def update(dct, bbox, row_id, *args, **kwargs):
    pts = np.array([[int(bbox[f"x{i}"]), int(bbox[f"y{i}"])] for i in range(1, 5)])
    x1 = pts[:, 0].min()
    y1 = pts[:, 1].min()
    x2 = pts[:, 0].max()
    y2 = pts[:, 1].max()
    if row_id in dct.keys():
        _x1, _y1, _x2, _y2 = dct[row_id]
        x1 = min(x1, _x1)
        x2 = max(x2, _x2)
        y1 = min(y1, _y1)
        y2 = max(y2, _y2)
    dct[row_id] = (x1, y1, x2, y2)


def show_dataset_bbox__format():
    for sample_idx in range(1000):
        img_path = f"{img_prefix}/receipt_{sample_idx:05}.png"
        json_path = f"{json_prefix}/receipt_{sample_idx:05}.json"
        img = cv2.imread(img_path)
        with open(json_path, "r") as ann_stream:
            ann_sample = json.load(ann_stream)

        row_id_to_clr = dict()
        row_id_to_bbox_price = dict()
        row_id_to_bbox_nm = dict()
        for sample in ann_sample["valid_line"]:
            for word_sample in sample["words"]:
                bbox_data = word_sample["quad"]

                color = (255, 0, 255)
                text = ""  # sample["category"]
                if sample["category"] in {"menu.price", "menu.nm"}:
                    text = sample["category"]
                    if sample["group_id"] not in row_id_to_clr.keys():
                        row_id_to_clr[sample["group_id"]] = (
                            random.randint(0, 255),
                            random.randint(0, 255),
                            random.randint(0, 255),
                        )
                    color = row_id_to_clr[sample["group_id"]]
                    dct = row_id_to_bbox_price  # if sample["category"] == "menu.price" else row_id_to_bbox_nm
                    update(dct, bbox_data, sample["group_id"])
        for row_id in row_id_to_clr.keys():
            if row_id in row_id_to_bbox_nm.keys():
                cv2.rectangle(
                    img, row_id_to_bbox_nm[row_id][0:2], row_id_to_bbox_nm[row_id][2:4], row_id_to_clr[row_id], 3
                )
            if row_id in row_id_to_bbox_price.keys():
                cv2.rectangle(
                    img, row_id_to_bbox_price[row_id][0:2], row_id_to_bbox_price[row_id][2:4], row_id_to_clr[row_id], 3
                )
        cv2.imshow("blabla", img)
        cv2.waitKey(0)


def get_ltwh(bbox):
    x1, y1, x2, y2 = bbox
    res = [int(i) for i in [x1, y1, x2 - x1, y2 - y1]]
    return res


def cord_to_coco():
    new_img_prefix = "/home/rmnv/dev/reciept-dissection/examples/true_train/image"
    ann_file_name = "/home/rmnv/dev/reciept-dissection/examples/true_train/annotations/train.json"

    result_ann = {
        "categories": [{"id": 1, "name": "position"}],
        "images": [],
        "annotations": [],
    }

    for sample_idx in tqdm(range(800)):
        img_path = f"{img_prefix}/receipt_{sample_idx:05}.png"
        json_path = f"{json_prefix}/receipt_{sample_idx:05}.json"
        img = cv2.imread(img_path)
        with open(json_path, "r") as ann_stream:
            ann_sample = json.load(ann_stream)

        row_id_to_bbox_price = dict()
        row_ids = set()
        row_id_to_bbox_nm = dict()

        for sample in ann_sample["valid_line"]:
            for word_sample in sample["words"]:
                bbox_data = word_sample["quad"]

                if sample["category"] in {"menu.price", "menu.nm"}:
                    if sample["group_id"] not in row_ids:
                        row_ids.add(sample["group_id"])
                    dct = row_id_to_bbox_price  # if sample["category"] == "menu.price" else row_id_to_bbox_nm
                    update(dct, bbox_data, sample["group_id"], img.shape[1], img.shape[0])
        result_ann["images"].append(
            {
                "id": sample_idx + 1,
                "width": float(img.shape[1]),
                "height": float(img.shape[0]),
                "file_name": f"image/receipt_{sample_idx:05}.jpg",
            }
        )
        for row_id in row_ids:
            # if row_id in row_id_to_bbox_nm.keys():
            #     result_ann["annotations"].append(
            #         {
            #             "id": len(result_ann["annotations"]) + 1,
            #             "category_id": 2,
            #             "image_id": sample_idx + 1,
            #             "bbox": get_ltwh(row_id_to_bbox_nm[row_id]),
            #         }
            #     )
            if row_id in row_id_to_bbox_price.keys():
                result_ann["annotations"].append(
                    {
                        "id": len(result_ann["annotations"]) + 1,
                        "category_id": 1,
                        "image_id": sample_idx + 1,
                        "bbox": get_ltwh(row_id_to_bbox_price[row_id]),
                    }
                )
        cv2.imwrite(f"{new_img_prefix}/receipt_{sample_idx:05}.jpg", img)
    with open(ann_file_name, "w") as stream:
        json.dump(result_ann, stream)

    print("SUCCESS!!!!!!!!!!! /-_- /-_- /-_-")


if __name__ == "__main__":
    # show_dataset_bbox__format()
    cord_to_coco()
