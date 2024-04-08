PREFIX := llm-vs-topic
IMAGE := $(PREFIX)-image

DOCKER-BUILD := docker build -t
DOCKER-RUN := docker run --rm --mount type=bind,source=$(shell pwd)/input/data,target=/user-zone/input-data --mount type=bind,source=$(shell pwd)/output/data,target=/user-zone/output-data --name

# Outils
help:
	cat README.md
build: Dockerfile
	$(DOCKER-BUILD) $(IMAGE) .
print-files: build
	$(DOCKER-RUN) $(PREFIX)-print-files $(IMAGE) print-files.py

# Protocole
## Pré-traitement
### TreeTagger
preprocessing-tree: build
	$(DOCKER-RUN) $(PREFIX)-preprocessing-tree $(IMAGE) preprocessing/treetagger.py
### Babelfy
preprocessing-babelfy-raw: build
	$(DOCKER-RUN) $(PREFIX)-preprocessing-babelfy-raw $(IMAGE) preprocessing/babelfy.py Raw
preprocessing-babelfy-rnn: build
	$(DOCKER-RUN) $(PREFIX)-preprocessing-babelfy-rnn $(IMAGE) preprocessing/babelfy.py RNNTagger
preprocessing-babelfy-tree: build
	$(DOCKER-RUN) $(PREFIX)-preprocessing-babelfy-tree $(IMAGE) preprocessing/babelfy.py TreeTagger

## Traitement
### CREA
### GPT
### LDA
### Llama2

## Evaluation
