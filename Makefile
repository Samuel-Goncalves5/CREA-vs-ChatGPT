PREFIX := llm-vs-topic
IMAGE := $(PREFIX)-image

DOCKER-BUILD := docker build -t
DOCKER-RUN := docker run --rm --mount type=bind,source=$(shell pwd)/input/data,target=/user-zone/input-data --mount type=bind,source=$(shell pwd)/output,target=/user-zone/output --name
LLAMA2 := --mount type=bind,source=$(shell pwd)/input/external/Llama-2,target=/user-zone/Llama-2

# Outils
help:
	cat README.md
build: Dockerfile
	$(DOCKER-BUILD) $(IMAGE) .
print-files: build
	$(DOCKER-RUN) $(PREFIX)-print-files $(IMAGE) print_files.py
generate-scenarios: build
	$(DOCKER-RUN) $(PREFIX)-generate-scenarios $(IMAGE) generate_scenarios.py

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
preprocessing-babelfy-rnn-punctuation: build
	$(DOCKER-RUN) $(PREFIX)-preprocessing-babelfy-rnn $(IMAGE) preprocessing/babelfy.py RNNTagger-punctuation
preprocessing-babelfy-rnn-no-punctuation: build
	$(DOCKER-RUN) $(PREFIX)-preprocessing-babelfy-rnn $(IMAGE) preprocessing/babelfy.py RNNTagger-no-punctuation
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
### Llama2
llama2: build
	$(DOCKER-RUN) $(PREFIX)-llama $(LLAMA2) $(IMAGE) Llama2.py
### LDA
lda: build
	$(DOCKER-RUN) $(PREFIX)-lda $(IMAGE) LDA.py
### ANTM
antm: build
	$(DOCKER-RUN) $(PREFIX)-antm $(IMAGE) ANTM.py
### Top2Vec
top2vec: build
	$(DOCKER-RUN) $(PREFIX)-top2vec $(IMAGE) Top2Vec.py
### BERTopic
bertopic: build
	$(DOCKER-RUN) $(PREFIX)-bertopic $(IMAGE) BERTopic.py
### GPT

## Evaluation
### Cohérences V et UMass
coherence-v-umass-eval: build
	$(DOCKER-RUN) $(PREFIX)-evaluation $(IMAGE) coherence.py

### Cohérence de sujets contextualisés (CTC)
ctc-eval: build
	$(DOCKER-RUN) $(PREFIX)-evaluation $(IMAGE) CTC.py

