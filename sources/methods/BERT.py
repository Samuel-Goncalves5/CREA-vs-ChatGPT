import numpy as np
import os
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import json

path = "../../input"


# Data pre-processing
texts_names = []
texts_contents = []
for filename in os.listdir(path + "/data/Raw"):
  f = open(path + "/data/Raw/" + filename, "r")
  lines = f.readlines()
  lines = [line.strip(" \n") for line in lines] # Remove \n and spaces
  lines = [line for line in lines if line]      # Remove empty lines
  texts_names.append(filename.split('.')[0])    # Remove .txt
  texts_contents.append(lines)
  f.close()
f = open(path+"/scenarios/scenarios.csv")
scenarios = f.readlines()
f.close()
scenarios = [scenario.strip("\n").split(';') for scenario in scenarios] # Remove \n and split the csv
scenarios = [(scenario[0], scenario[1:]) for scenario in scenarios]     # Isolate name from texts
texts_per_scenario = [(scenario, [texts_contents[i] for i in range(len(texts_names)) if texts_names[i] in content]) for scenario, content in scenarios]
texts_per_scenario = [(scenario, [sentence for text in content for sentence in text]) for scenario, content in texts_per_scenario]

# Obtaining document embeddings
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
texts_embeddings = [model.encode(lines) for lines in texts_contents]
embedding_per_scenario = []
for scenario, content in scenarios:
  embedding = [texts_embeddings[i] for i in range(len(texts_names)) if texts_names[i] in content]
  embedding_per_scenario.append((scenario, np.vstack(embedding)))

# UMAP
umap_embedding_per_scenario = [(scenario, umap.UMAP().fit_transform(content)) for scenario, content in embedding_per_scenario]

# HDBSAN
labels_per_scenario = [(scenario, hdbscan.HDBSCAN(min_cluster_size=10).fit_predict(content)) for scenario, content in umap_embedding_per_scenario]
per_labels_sentences = []

for i in range(len(scenarios)):
  scenario, content = labels_per_scenario[i]
  _, sentences = texts_per_scenario[i]
  label_sentences = []
  label = 0
  while len(content[content == label]) != 0:
    args = np.argwhere(content == label).reshape(-1)
    label_sentences.append([sentences[j] for j in args])
    label += 1
  per_labels_sentences.append((scenario, label_sentences))

with open('raw_per_labels_sentences.json', 'w') as file:
  json.dump(per_labels_sentences, file, indent=4)

with open('raw_per_labels_sentences.json', 'r') as file:
    per_labels_sentences = json.load(file)

with open('raw_labels_per_scenario.json', 'w') as file:
  json.dump(labels_per_scenario, file, indent=4)

with open('raw_labels_per_scenario.json', 'r') as file:
    labels_per_scenario = json.load(file)

print(per_labels_sentences[0][0], len(per_labels_sentences[0][1]))
for i in range(len(per_labels_sentences[0][1])):
  print(i, per_labels_sentences[0][1][i])

# Visualisation
mappers = []
for i in range(len(labels_per_scenario)):
    labels = labels_per_scenario[i][1]
    mapper = umap.UMAP().fit(embedding_per_scenario[i][1][labels != -1])
    mappers.append(mapper)

with open('raw_mappers.json', 'w') as file:
  json.dump(mappers, file, indent=4)

with open('raw_mappers.json', 'r') as file:
    mappers = json.load(file)

labels = labels_per_scenario[i][1]
umap.plot.points(mappers[0], labels=labels[labels != -1])