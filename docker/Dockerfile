FROM python:3.8.10

RUN apt-get update \
    && apt-get install -y git
# RUN apt -y install python3-pip


RUN git clone https://github.com/nekiynekit/reciept-dissection.git
RUN  cd reciept-dissection/ && pip install -r requirements.txt

ARG TOKEN
RUN export TOKEN=$TOKEN

WORKDIR /reciept-dissection
