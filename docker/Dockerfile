FROM python:3.8.10

RUN apt-get update \
    && apt-get install -y git
# RUN apt -y install python3-pip
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


RUN git clone https://github.com/nekiynekit/reciept-dissection.git
RUN cd reciept-dissection/ && pip install -r requirements.txt

ARG TOKEN
ENV export TOKEN=${TOKEN}

WORKDIR /reciept-dissection

EXPOSE 8000

ENTRYPOINT gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker
