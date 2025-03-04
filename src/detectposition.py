import random

import cv2
import easyocr
import numpy as np
import PIL
import PIL.Image
import regex as re
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
    easyocr_model = easyocr.Reader(["en"])
    return model, pipe, easyocr_model


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
        return intersection

    for i, pb in enumerate(price_boxes):
        for j, nb in enumerate(nm_boxes):
            cost_matrix[i, j] = -metric(pb, nb)

    p_indicies, n_indicies = linear_sum_assignment(cost_matrix)
    valid_output_pairs = [(price_boxes[p_i], nm_boxes[n_i]) for p_i, n_i in zip(p_indicies, n_indicies)]
    valid_output_pairs = sorted(valid_output_pairs, key=lambda box: box[0][1])
    valid_output_pairs = [(p, n) for (p, n) in valid_output_pairs if metric(p, n) > 1]

    return valid_output_pairs


def get_sec(img, xyxy, pad_x=0.99, pad_y=3):
    x1, y1, x2, y2 = xyxy
    if pad_x < 1 and pad_x > 0:
        pad_x = int((x2 - x1) / 2)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img.shape[1] - 1, x2 + pad_x)
    y2 = min(img.shape[0] - 1, y2 + pad_y)
    return img[y1:y2, x1:x2]


def idxs_to_text(ocr_model, easyocr_model, img, valid_pairs):
    response_json = list()
    sectors = list()
    prices = list()
    for xyxy_p, xyxy_n in valid_pairs:
        nm_sec = PIL.Image.fromarray(cv2.cvtColor(get_sec(img, xyxy_n, pad_x=0, pad_y=0), cv2.COLOR_BGR2RGB))
        price = easyocr_model.readtext(cv2.cvtColor(get_sec(img, xyxy_p), cv2.COLOR_BGR2RGB))[0][1]
        sectors.append(nm_sec)
        prices.append(price)
    model_out = ocr_model(sectors)

    for nm_sample, price in zip(model_out, prices):
        nm_text = nm_sample[0]["generated_text"]
        p_text = price
        response_json.append([nm_text, p_text])
    return response_json


def draw_img(img, valid_pairs, boxes):
    img = img.copy()
    for p, n in valid_pairs:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for box, pad_x, pad_y in zip([p, n], [0.99, 0], [3, 0]):
            x1, y1, x2, y2 = box
            if pad_x < 1 and pad_x > 0:
                pad_x = int((x2 - x1) / 2)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(img.shape[1] - 1, x2 + pad_x)
            y2 = min(img.shape[0] - 1, y2 + pad_y)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    cv2.imwrite("final.jpg", img)


def filter_position(bill):
    valid_bill = list()
    for nm_text, p_text in bill:
        print(f"Orig: {nm_text} price is {p_text}")
        p_text = "".join(p_text.replace("Q", "0").replace("O", "0").split(" "))
        if re.fullmatch(r"^[0123456789,. \t]+$", p_text) is None:
            print(f"    -_- drop price [{p_text}], don't match")
            continue
        p_text = p_text.split(",")[0].split(".")[0]
        try:
            p_text = int(p_text)
        except:
            continue
        valid_bill.append([nm_text, str(p_text)])
    return valid_bill


def cook_the_bill(yolo_model, ocr_model, easyocr_model, img, yolo_conf=0.3):
    print(f"RUNNING MODELS")
    yolo_output = yolo_model.predict(img, conf=yolo_conf, imgsz=1280, iou=0.3)[0].boxes
    print(f"MATCHING PAIRS")
    valid_pairs = calculate_matching(yolo_output)
    print(f"DRAW")
    draw_img(img, valid_pairs, yolo_output.xyxy)
    print(f"PREPARING..")
    bill = idxs_to_text(ocr_model, easyocr_model, img, valid_pairs)
    print(f"CLEAR THE SUMS...")
    bill = filter_position(bill)
    return bill


if __name__ == "__main__":
    pass
