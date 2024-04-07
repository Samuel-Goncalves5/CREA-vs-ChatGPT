FROM python:3.10.13-alpine3.19

WORKDIR user-zone/

COPY requirements.txt .
RUN apk add build-base && pip install -r requirements.txt

COPY sources/methods sources/utils .
COPY sources/preprocessing/* input/external/pybabelfy input/external/nltk preprocessing
COPY input/data input-data
COPY output/data output-data

RUN ["python", "preprocessing/nltk-download.py"]
ENTRYPOINT ["python"]