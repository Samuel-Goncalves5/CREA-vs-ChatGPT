PREFIX := llm-vs-topic
IMAGE := $(PREFIX)-image

DOCKER-BUILD := docker build -t
DOCKER-RUN := docker run --rm --mount type=bind,source=$(shell pwd)/input/data,target=/user-zone/data --name

help:
	cat README.md
print-files: build
	$(DOCKER-RUN) $(PREFIX)-print-files $(IMAGE) print-files.py
preprocessing-babelfy: build
	$(DOCKER-RUN) $(PREFIX)-preprocessing-babelfy $(IMAGE) preprocessing/babelfy.py
build: Dockerfile
	$(DOCKER-BUILD) $(IMAGE) .
