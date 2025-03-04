import os
import time
from threading import Thread

import cv2
import numpy as np
import PIL
import requests
import telebot as tb
from fastapi import FastAPI

from src.detectposition import cook_the_bill, prepare_modules

token = os.environ.get("TOKEN")
bot = tb.TeleBot(token)

task_pool = dict()


app = FastAPI()

is_nns_ready = False
yolo_model, ocr_model, easyocr_model = None, None, None


@app.get("/")
def start_page():
    return "nns ready!"


@app.get("/run_task")
def get_task(file_id):
    global task_pool
    task_id = 0 if len(task_pool) == 0 else max(task_pool.keys()) + 1
    task = Thread(target=run_pipeline, args=[file_id, task_id])
    task.start()
    task_pool[task_id] = [task, None, time.time()]
    task_pool = {
        task_key: [thr, resp, start_time]
        for task_key, [thr, resp, start_time] in task_pool.items()
        if time.time() - start_time < 300
    }
    return str(task_id)


@app.get("/get_bill")
def check_task(task_id):
    global task_pool
    print(task_pool)
    if int(task_id) in task_pool.keys() and not task_pool[int(task_id)][0].isAlive():
        print("REMOVE FROM TASK_POOL")
        response = task_pool[int(task_id)][1]
        # task_pool.pop(int(task_id))
        return response
    return "not ready"


def run_pipeline(file_id, task_id):
    print(f"START PIPELINE, {file_id=}, {task_id=}")
    global yolo_model, ocr_model, is_nns_ready, bot, token, task_pool, easyocr_model

    # Initialize models
    if not is_nns_ready:
        yolo_model, ocr_model, easyocr_model = prepare_modules()
        is_nns_ready = True

    # Read and convert image
    file_path = bot.get_file(file_id).file_path
    img_np = np.asarray(
        PIL.Image.open(requests.get(f"https://api.telegram.org/file/bot{token}/{file_path}", stream=True).raw)
    )

    # Log image and process bill
    cv2.imwrite("image.jpg", img_np)
    response = cook_the_bill(yolo_model, ocr_model, easyocr_model, img_np)

    print("-----------------READY------------------")
    for n, p in response:
        print(n, "\n", p)
        print()
    print("----------------------------------------")

    task_pool[task_id][1] = response
    print(task_pool)
    print(type(task_id), task_id)
