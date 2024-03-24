FROM python:3.10.13-alpine3.19

WORKDIR user-zone/

COPY requirements.txt .
RUN apk add build-base && pip install -r requirements.txt

COPY sources/methods sources/utils .
COPY sources/preprocessing/Babelfy input/external/pybabelfy sources/preprocessing/RNNTagger sources/preprocessing/TreeTagger preprocessing
COPY input/data data

ENTRYPOINT ["python"]
