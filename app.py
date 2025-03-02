import asyncio
import os

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
yolo_model, ocr_model = None, None


@app.get("/")
async def start_page():
    return "nns ready!"


@app.get("/run_task")
async def get_task(file_id):
    task_id = 0 if len(task_pool) == 0 else max(task_pool.keys()) + 1
    asyncio.create_task(run_pipeline(file_id, task_id))
    return str(task_id)


@app.get("/get_bill")
async def check_task(task_id):
    global task_pool
    print(task_pool)
    if int(task_id) in task_pool.keys():
        print("REMOVE FROM TASK_POOL")
        response = task_pool[int(task_id)]
        # task_pool.pop(int(task_id))
        return response
    return "not ready"


async def run_pipeline(file_id, task_id):
    print(f"START PIPELINE, {file_id=}, {task_id=}")
    await asyncio.sleep(0)
    global yolo_model, ocr_model, is_nns_ready, bot, token, task_pool

    # Initialize models
    if not is_nns_ready:
        yolo_model, ocr_model = prepare_modules()
        is_nns_ready = True

    # Read and convert image
    file_path = bot.get_file(file_id).file_path
    img_np = np.asarray(
        PIL.Image.open(requests.get(f"https://api.telegram.org/file/bot{token}/{file_path}", stream=True).raw)
    )

    # Log image and process bill
    cv2.imwrite("image.jpg", img_np)
    response = cook_the_bill(yolo_model, ocr_model, img_np)

    print("-----------------READY------------------")
    for n, p in response:
        print(n, "\n", p)
        print()
    print("----------------------------------------")

    task_pool[task_id] = response
    print(task_pool)
    print(type(task_id), task_id)
