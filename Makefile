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
### RNNTagger
preprocessing-rnn: build
	$(DOCKER-RUN) $(PREFIX)-preprocessing-rnn $(IMAGE) preprocessing/rnntagger.py
### TreeTagger
preprocessing-tree: build
	$(DOCKER-RUN) $(PREFIX)-preprocessing-tree $(IMAGE) preprocessing/treetagger.py
### Babelfy
preprocessing-babelfy-raw: build
	$(DOCKER-RUN) $(PREFIX)-preprocessing-babelfy-raw $(IMAGE) preprocessing/babelfy.py Raw
preprocessing-babelfy-rnn: build
	$(DOCKER-RUN) $(PREFIX)-preprocessing-babelfy-rnn $(IMAGE) preprocessing/babelfy.py RNNTagger
preprocessing-babelfy-tree-no-stop: build
	$(DOCKER-RUN) $(PREFIX)-preprocessing-babelfy-tree-no-stop $(IMAGE) preprocessing/babelfy.py TreeTagger-no-stop
preprocessing-babelfy-tree-stop-word: build
	$(DOCKER-RUN) $(PREFIX)-preprocessing-babelfy-tree-stop-word $(IMAGE) preprocessing/babelfy.py TreeTagger-stop-word
preprocessing-babelfy-tree-stop-class: build
	$(DOCKER-RUN) $(PREFIX)-preprocessing-babelfy-tree-stop-class $(IMAGE) preprocessing/babelfy.py TreeTagger-stop-class

## Traitement
### CREA
crea: build
	$(DOCKER-RUN) $(PREFIX)-crea $(IMAGE) CREA.py
### GPT
### LDA
lda: build
	$(DOCKER-RUN) $(PREFIX)-lda $(IMAGE) LDA.py
### Llama2

## Evaluation
