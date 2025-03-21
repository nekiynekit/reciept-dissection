# reciept-dissection
Repo with OCR reciept dissection node for tg-bot. Parsing document with YOLO, easyOCR and some regexp. For russian naming extraction there are `raxtemur/trocr-base-ru` model from huggingface.

# Running with docker
```bash
docker build -t ocr docker/
docker run --rm -it --net=host -p 8000:8000 -e TOKEN=... ocr
```

# Pipeline scheme
At first stage custom-trained YOLO detect boxes with text and classify them onto `prices` and `positions`. Next we calculate metric between these classes like horizontal intersection (across boxes Y-positions). Put it into cost matrix and calculate maximum cost assignment (Hungarian algorithm). Now send pairs every position and price into OCR model to extract actual price and position name. For price extraction here I use `easyOCR` library, that lightweight and fast enough, and for russian-labeled reciept position `raxtemur/trocr-base-ru` model using. Next prices should befiltered by regular expression, cast to integers and sent to output.

# Backend 
FastAPI used as framework. Use `run_task(file_id)->task_id` for initialize processing image and recieve result by `get_bill(task_id)->...`

# YOLO training
YOLO from ultralytics trained on a CORD dataset. All scripts for visualize and preprocessing dataset are in `src/cord_overview.py` file.

# Pipe example
<p align="center">
  <img src="https://raw.githubusercontent.com/nekiynekit/reciept-dissection/main/media/final.jpg" title="Boxes render after processing">
</p>

### Output:
```
Белое золото 0,5л 
 3000

Кувшин клюкв морса 
 1380

Строганина муксун 5020 
 1040

Строганина нельга 5020 
 1420

Северный бот 
 3190

Огурцы малосол смелом 
 590

Грузии Селье 
 860

Пельмени Порт-Артур 
 2980
```
