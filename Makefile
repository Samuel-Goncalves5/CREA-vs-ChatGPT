PREFIX := llm-vs-topic
IMAGE := $(PREFIX)-image

DOCKER-BUILD := docker build -t
DOCKER-RUN := docker run --rm --mount type=bind,source=$(shell pwd)/input/data,target=/user-zone/input-data --mount type=bind,source=$(shell pwd)/output/data,target=/user-zone/output-data --name

help:
	cat README.md
print-files: build
	$(DOCKER-RUN) $(PREFIX)-print-files $(IMAGE) print-files.py
preprocessing-babelfy-raw: build
	$(DOCKER-RUN) $(PREFIX)-preprocessing-babelfy-raw $(IMAGE) preprocessing/babelfy.py Raw
preprocessing-babelfy-rnn: build
	$(DOCKER-RUN) $(PREFIX)-preprocessing-babelfy-rnn $(IMAGE) preprocessing/babelfy.py RNNTagger
preprocessing-babelfy-tree: build
	$(DOCKER-RUN) $(PREFIX)-preprocessing-babelfy-tree $(IMAGE) preprocessing/babelfy.py TreeTagger
build: Dockerfile
	$(DOCKER-BUILD) $(IMAGE) .
