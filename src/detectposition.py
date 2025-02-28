import random

import cv2
import numpy as np
import PIL
import PIL.Image
from scipy.optimize import linear_sum_assignment
from transformers import pipeline
from ultralytics import YOLO


def play_with_fixed_image():
    model = YOLO("weights/yolov11nano-cord-divide.pt")
    model_output = model.predict("examples/reciept.jpg", conf=0.4)
    img = cv2.imread("examples/reciept.jpg")

    pipe = pipeline("image-to-text", model="raxtemur/trocr-base-ru")
    img_copy = img.copy()

    for (x1, y1, x2, y2), conf, label in zip(
        model_output[0].boxes.xyxy, model_output[0].boxes.conf, model_output[0].boxes.cls
    ):
        x1, y1, x2, y2 = [int(i) for i in [x1, y1, x2, y2]]
        if int(label) == 0:
            # price
            color = (0, 0, 0)
        else:
            color = (255, 0, 0)
        copy_img = img.copy()

        sector = img[y1:y2, x1:x2]
        text = pipe(PIL.Image.fromarray(sector, "RGB"))
        print(text)

        cv2.rectangle(copy_img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        cv2.putText(copy_img, str(conf), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, cv2.LINE_4)
        cv2.putText(img_copy, str(conf), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, cv2.LINE_4)
        cv2.imshow("fullimg", copy_img)
        # cv2.imshow("sector", sector)
        cv2.waitKey(0)
    cv2.imshow("finally", img_copy)
    cv2.waitKey(0)


def prepare_modules(weights="weights/yolov11nano-cord-divide.pt"):
    model = YOLO(weights)
    pipe = pipeline("image-to-text", model="raxtemur/trocr-base-ru")
    return model, pipe


def calculate_matching(yolo_outputs):
    price_boxes = list(filter(lambda tpl: int(tpl[1]) == 0, zip(yolo_outputs.xyxy, yolo_outputs.cls)))
    price_boxes = [np.array(pb[0]).astype(np.uint64).tolist() for pb in price_boxes]
    nm_boxes = list(filter(lambda tpl: int(tpl[1]) == 1, zip(yolo_outputs.xyxy, yolo_outputs.cls)))
    nm_boxes = [np.array(nb[0]).astype(np.uint64).tolist() for nb in nm_boxes]
    cost_matrix = np.zeros((len(price_boxes), len(nm_boxes)))

    def metric(price_box, nm_box):
        l1, r1 = price_box[1], price_box[3]
        l2, r2 = nm_box[1], nm_box[3]
        intersection = max(0, min(r1, r2) - max(l1, l2))
        print(f"between {[l1, r1]} and {[l2, r2]} metric is {intersection}")
        return intersection

    for i, pb in enumerate(price_boxes):
        for j, nb in enumerate(nm_boxes):
            cost_matrix[i, j] = -metric(pb, nb)

    print("cost mat = ", cost_matrix)
    p_indicies, n_indicies = linear_sum_assignment(cost_matrix)
    valid_output_pairs = [(price_boxes[p_i], nm_boxes[n_i]) for p_i, n_i in zip(p_indicies, n_indicies)]
    valid_output_pairs = [(p, n) for (p, n) in valid_output_pairs if metric(p, n) > 1]
    return valid_output_pairs


def get_sec(img, xyxy):
    x1, y1, x2, y2 = xyxy
    return img[y1:y2, x1:x2]


def idxs_to_text(ocr_model, img, valid_pairs):
    response_json = list()
    for xyxy_p, xyxy_n in valid_pairs:
        print(xyxy_n, xyxy_p)
        nm_text = ocr_model(PIL.Image.fromarray(get_sec(img, xyxy_n)))[0]["generated_text"]
        p_text = ocr_model(PIL.Image.fromarray(get_sec(img, xyxy_p)))[0]["generated_text"]
        response_json.append([nm_text, p_text])
    return response_json


def draw_img(img, valid_pairs, boxes):
    img = img.copy()
    for p, n in valid_pairs:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for box in [p, n]:
            print(box)
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    cv2.imwrite("final.jpg", img)


def cook_the_bill(yolo_model, ocr_model, img, yolo_conf=0.3):
    yolo_output = yolo_model.predict(img, conf=yolo_conf, imgsz=1280, iou=0.3)[0].boxes
    valid_pairs = calculate_matching(yolo_output)
    draw_img(img, valid_pairs, yolo_output.xyxy)
    bill = idxs_to_text(ocr_model, img, valid_pairs)
    return bill


if __name__ == "__main__":
    pass
