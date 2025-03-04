# reciept-dissection
Repo with OCR reciept dissection node for tg-bot. Parsing document with YOLO, easyOCR and some regexp. For russian naming extraction there are `raxtemur/trocr-base-ru` model from huggingface.

# Running with docker
set `TOKEN` in .env file for your tg-bot for reading `file_id` from Telegram
```bash
docker build -t reciept-dissection docker/
```

# Pipeline scheme
At first stage custom-trained YOLO detect boxes with text and classify them onto `prices` and `positions`. Next we calculate metric between these classes like horizontal intersection (across boxes Y-positions). Put it into cost matrix and calculate maximum cost assignment (Hungarian algorithm). Now send pairs every position and price into OCR model to extract actual price and position name. For price extraction here I use `easyOCR` library, that lightweight and fast enough, and for russian-labeled reciept position `raxtemur/trocr-base-ru` model using. Next prices should befiltered by regular expression, cast to integers and sent to output.

# Backend 
FastAPI used as framework. Use `run_task(file_id)->task_id` for initialize processing image and recieve result by `get_bill(task_id)->...`

# YOLO training
YOLO from ultralytics trained on a CORD dataset. All scripts for visualize and preprocessing dataset are in `src/cord_overview.py` file.

# Pipe example
<p align="center">
  <img src="your_relative_path_here" width="350" title="hover text">
  <img src="your_relative_path_here_number_2_large_name" width="350" alt="accessibility text">
</p>