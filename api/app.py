import os

import cv2
import numpy as np
import PIL
import requests
import telebot as tb
from flask import Flask, request

from src.detectposition import cook_the_bill, prepare_modules

token = os.environ.get("TOKEN")
bot = tb.TeleBot(token)


app = Flask(__name__)

is_nns_ready = False

yolo_model, ocr_model = None, None


@app.route("/", methods=["GET", "POST"])
def start_page():
    global yolo_model, ocr_model, is_nns_ready, bot

    bot.register_message_handler(photo_id, pass_bot=True)

    bot.infinity_polling()
    return "nns ready!"


@app.route("/get_bill", methods=["GET", "POST"])
def get_task():
    global yolo_model, ocr_model, is_nns_ready, bot, token
    if not is_nns_ready:
        yolo_model, ocr_model = prepare_modules()
    file_id = request.args.get("file_id")
    file_path = bot.get_file(file_id).file_path
    img_np = np.asarray(
        PIL.Image.open(requests.get(f"https://api.telegram.org/file/bot{token}/{file_path}", stream=True).raw)
    )
    cv2.imwrite("image.jpg", img_np)
    response = cook_the_bill(yolo_model, ocr_model, img_np)
    for n, p in response:
        print(n, '\n', p)
        print()
    return response


@bot.message_handler(content_types=["photo"])
def photo_id(message):
    photo = max(message.photo, key=lambda x: x.height)
    print(photo.file_id)
