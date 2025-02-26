import json
import random

import click
import cv2
import numpy as np

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


for sample_idx in range(1000):
    img_path = f"{img_prefix}/receipt_{sample_idx:05}.png"
    json_path = f"{json_prefix}/receipt_{sample_idx:05}.json"
    img = cv2.imread(img_path)
    with open(json_path, "r") as ann_stream:
        ann_sample = json.load(ann_stream)

    row_id_to_clr = dict()
    for sample in ann_sample["valid_line"]:
        for word_sample in sample["words"]:
            bbox_data = word_sample["quad"]

            color = (255, 0, 255)
            text = ""  # sample["category"]
            if sample["category"] in {"menu.price", "menu.nm"}:
                text = sample["category"]
                if word_sample["row_id"] not in row_id_to_clr.keys():
                    row_id_to_clr[word_sample["row_id"]] = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255),
                    )
                color = row_id_to_clr[word_sample["row_id"]]
            draw_bbos(img, bbox_data, color, text)
    cv2.imshow("blabla", img)
    cv2.waitKey(0)
