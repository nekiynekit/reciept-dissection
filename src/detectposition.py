import cv2
import PIL
import PIL.Image
from transformers import pipeline
from ultralytics import YOLO

model = YOLO("weights/yolov11nano-cord.pt")
model_output = model.predict("examples/reciept.jpg", conf=0.55)
img = cv2.imread("examples/reciept.jpg")


pipe = pipeline("image-to-text", model="raxtemur/trocr-base-ru")

for (x1, y1, x2, y2), conf, label in zip(
    model_output[0].boxes.xyxy, model_output[0].boxes.conf, model_output[0].boxes.cls
):
    x1, y1, x2, y2 = [int(i) for i in [x1, y1, x2, y2]]
    if label == 0:
        # price
        color = (0, 0, 0)
    else:
        color = (255, 0, 0)
    copy_img = img.copy()

    sector = img[y1:y2, x1:x2]
    text = pipe(PIL.Image.fromarray(sector, "RGB"))
    print(text)

    cv2.rectangle(copy_img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(copy_img, str(conf), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, cv2.LINE_4)
    cv2.imshow("fullimg", img)
    cv2.imshow("sector", sector)
    cv2.waitKey(0)
