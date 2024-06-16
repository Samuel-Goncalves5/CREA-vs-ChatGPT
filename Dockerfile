FROM python:3.10.13

WORKDIR user-zone/

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY sources/methods sources/evaluations sources/utils .
COPY sources/preprocessing/* input/external/pybabelfy input/external/nltk preprocessing
COPY input/external/TreeTagger preprocessing/TreeTagger
COPY input/external/RNNTagger preprocessing/RNNTagger
COPY input/data input-data
COPY output/data output-data

RUN ["python", "preprocessing/nltk-download.py"]
ENTRYPOINT ["python"]