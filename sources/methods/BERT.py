import numpy as np
import os
from sentence_transformers import SentenceTransformer
import umap
import umap.plot
import matplotlib.pyplot as plt
import hdbscan
import json
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

inputPath = "../../input/"
outputPath = "./BERT_JSONs/"
mode = ["raw"]

###############
### IMPORTs ###
###############

def importRawFiles():
    texts_names = []
    texts_contents = []
    for filename in os.listdir(inputPath + "/data/Raw"):
        f = open(inputPath + "/data/Raw/" + filename, "r")
        lines = f.readlines()
        lines = [line.strip(" \n") for line in lines] # Remove \n and external spaces
        lines = [line for line in lines if line]      # Remove empty lines
        texts_names.append(filename.split('.')[0])    # Remove .txt
        texts_contents.append(lines)
        f.close()
    return texts_names, texts_contents

def importBabelEquivalentFiles():
    dictionaryBabelfy = {}
    for filename in os.listdir(inputPath + "/data/Raw+Babelfy/dictionary"):
        f = open(inputPath + "/data/Raw+Babelfy/dictionary/" + filename, "r")
        lines = f.readlines()
        lines = [line.strip("\n").split(";") for line in lines]
        for k,v in lines:
            dictionaryBabelfy[k] = v
        f.close()
    
    texts_names = []
    texts_contents = []
    subpath = "/data/Raw+Babelfy/equivalent/"
    for filename in os.listdir(inputPath + subpath):
        f = open(inputPath + subpath + filename, "r")
        lines = f.readlines()
        lines = [line.strip(" \n") for line in lines]       # Remove \n and external spaces
        lines = [dictionaryBabelfy[line] for line in lines] # Reconstruct the notions
        texts_names.append(filename.split('.')[0])          # Remove .txt
        texts_contents.append(lines)
        f.close()
    return texts_names, texts_contents

imports = {'raw': importRawFiles, 'babel': importBabelEquivalentFiles}

#################
### SCENARIOs ###
#################

def createScenarios(texts_names, texts_contents):
    f = open(inputPath + "/scenarios/scenarios.csv")
    scenarios = f.readlines()
    scenarios = [scenario.strip("\n").split(';') for scenario in scenarios] # Remove \n and split the csv
    scenarios = [(scenario[0], scenario[1:]) for scenario in scenarios]     # Isolate name from texts
    f.close()

    texts_per_scenario = [(scenario, [texts_contents[i] for i in range(len(texts_names)) if texts_names[i] in content]) for scenario, content in scenarios]
    sentences_per_scenario = [(scenario, [sentence for text in content for sentence in text]) for scenario, content in texts_per_scenario]

    return scenarios, texts_per_scenario, sentences_per_scenario

def embeddingsPerScenarios(embeddings, scenarios):
    embedding_per_scenario = []
    for scenario, content in scenarios:
        embedding = [embeddings[i] for i in range(len(texts_names)) if texts_names[i] in content]
        embedding = [text for name, text in embedding]
        embedding_per_scenario.append((scenario, np.vstack(embedding)))
    return embedding_per_scenario

#######################
### BERT EMBEDDINGS ###
#######################

def createEmbeddings(texts_names, texts_contents):
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    texts_embeddings = []
    for i in range(len(texts_names)):
        name = texts_names[i]
        content = texts_contents[i]
        texts_embeddings.append((name, [model.encode(lines) for lines in content]))
    return texts_embeddings

#######################
### UMAP EMBEDDINGS ###
#######################

def createUmapEmbeddings(embeddings_per_scenario):
    umap_embeddings_per_scenario = []
    for scenario, content in embeddings_per_scenario:
        umap_embeddings = umap.UMAP().fit_transform(content)
        umap_embeddings_per_scenario.append((scenario, umap_embeddings))
    return umap_embeddings_per_scenario

#####################
### HDBSAN LABELS ###
#####################

def createHdbsanLabels(umap_embeddings_per_scenario, min_cluster_size):
    labels_per_scenario = []
    for i in range(len(umap_embeddings_per_scenario)):
        scenario, content = umap_embeddings_per_scenario[i]
        labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size[i]).fit_predict(content)
        labels_per_scenario.append((scenario, labels))
    return labels_per_scenario 

def assignLabels(labels_per_scenario, sentences_per_scenario):
    per_label_sentences = []
    for i in range(len(labels_per_scenario)):
        scenario, content = labels_per_scenario[i]
        _, sentences = sentences_per_scenario[i]
        
        label_sentences = []
        label_iter = 0
        while len(content[content == label_iter]) != 0:
            args = np.argwhere(content == label_iter).reshape(-1)
            label_sentences.append([sentences[j] for j in args])
            label_iter += 1
        
        per_label_sentences.append((scenario, label_sentences))
    
    return per_label_sentences

##########################
### UMAP VISUALISATION ###
##########################

def createUmapVisualisations(labels_per_scenario, embeddings_per_scenario, path):
    for i in range(len(labels_per_scenario)):
        scenario, labels = labels_per_scenario[i]

        if len(labels[labels != -1]) == 0:
          continue

        mapper = umap.UMAP().fit(embeddings_per_scenario[i][1][labels != -1])
        _, ax = plt.subplots()
        umap.plot.points(mapper, labels=labels[labels != -1], ax=ax)
        plt.savefig(path + "_" + scenario + ".png", dpi=300)

########################
### TOPIC PRODUCTION ###
########################

def createTopics(per_label_sentences, max_size):
    topics_per_scenario = []

    for scenario, sentence_classes in per_label_sentences:
        topics = []
        for sentence_class in sentence_classes:
            occurences = {}
            for sentence in sentence_class:
                for word in sentence.split(' '):
                    if word in occurences:
                        occurences[word] += 1
                    else:
                        occurences[word] = 1
            topic = sorted(occurences.items(), key=lambda x: x[1], reverse=True)[:max_size]
            topic = [word for word,_ in topic]
            topics.append(topic)
        topics_per_scenario.append((scenario, topics))

    return topics_per_scenario

##########################################################
####################### COHERENCE ########################
##########################################################

def coherence_func(outputFile, topics, corpus, texts, dictionary, coherence):
    cm = CoherenceModel(topics=topics, corpus=corpus, dictionary=dictionary, texts=texts, coherence=coherence)
    coherence = cm.get_coherence()
    per_topic = cm.get_coherence_per_topic()
    per_topic = [str(topic) for topic in per_topic]
    per_topic = ";".join(per_topic)

    f = open(outputFile, "w")
    f.write("coherence;" + str(coherence) + "\n" + "coherence per topic;" + per_topic)
    f.close()

#############
### TOOLS ###
#############

def jsonPrepare(value):
    if isinstance(value, np.ndarray):
        return ["__NP_NDARRAY__", jsonPrepare(list(value))]
    elif isinstance(value, (list, tuple)):
        return [jsonPrepare(v) for v in value]
    elif isinstance(value, dict):
        return {k: jsonPrepare(v) for k, v in value.items()}
    elif isinstance(value, np.float32):
        return ["__NP_FLOAT32__", float(value)]
    elif isinstance(value, np.float64):
        return ["__NP_FLOAT64__", float(value)]
    elif isinstance(value, np.int32):
        return ["__NP_INT32__", int(value)]
    elif isinstance(value, np.int64):
        return ["__NP_INT64__", int(value)]
    else:
        return value

def jsonRead(value):
    if isinstance(value, (list, tuple)):
        if (len(value) == 2 and value[0] == "__NP_NDARRAY__"):
            return np.array([jsonRead(v) for v in value[1]])
        elif (len(value) == 2 and value[0] == "__NP_FLOAT32__"):
            return np.float32(value[1])
        elif (len(value) == 2 and value[0] == "__NP_FLOAT64__"):
            return np.float64(value[1])
        elif (len(value) == 2 and value[0] == "__NP_INT32__"):
            return np.int32(value[1])
        elif (len(value) == 2 and value[0] == "__NP_INT64__"):
            return np.int64(value[1])
        else:
            return [jsonRead(v) for v in value]
    elif isinstance(value, dict):
        return {k: jsonRead(v) for k, v in value.items()}
    else:
        return value

def memoisation(path, name, function):
    file_path = path + "/" + name + ".json"
    if name + ".json" in os.listdir(path):
        f = open(file_path, "r")
        val = jsonRead(json.load(f))
        f.close()
    else:
        val = function(None)
        f = open(file_path, "w")
        json.dump(jsonPrepare(val), f, indent=4)
        f.close()

    return val

###################
### APPLICATION ###
###################

if __name__ == '__main__':
    print("import...")
    texts_names, texts_contents = imports[mode[0]]()

    print("scenarios...")
    scenarios, texts_per_scenario, sentences_per_scenario = createScenarios(texts_names, texts_contents)

    print("bert...")
    embeddings = memoisation(outputPath + mode[0], "embeddings",
                             lambda _ : createEmbeddings(texts_names, texts_contents))
    embeddings_per_scenario = embeddingsPerScenarios(embeddings, scenarios)

    print("umap...")
    umap_embeddings_per_scenario = memoisation(outputPath + mode[0], "umap",
                                               lambda _ : createUmapEmbeddings(embeddings_per_scenario))

    print("cluster preparation...")
    lengths = [len(content) for _, content in umap_embeddings_per_scenario]
    cluster_sizes = [("10",  [10                 for _      in lengths]),
                     ("20",  [20                 for _      in lengths]),
                     ("01p", [    length // 1000 for length in lengths]),
                     ("05p", [    length // 200  for length in lengths]),
                     ("1p",  [    length // 100  for length in lengths]),
                     ("2p",  [2 * length // 100  for length in lengths]),
                     ("5p",  [5 * length // 100  for length in lengths])]

    print("hdbscan...")
    for name, cluster_size in cluster_sizes:
        print("hdbscan - " + name + "...")
        labels_per_scenario = memoisation(outputPath + mode[0] + "/hdbscan", name,
                                          lambda _ : createHdbsanLabels(umap_embeddings_per_scenario, cluster_size))

        print("hdbscan - " + name + " visualisation...")
        path = outputPath + mode[0] + "/hdbscan/" + name
        createUmapVisualisations(labels_per_scenario, embeddings_per_scenario, path)

        print("hdbscan - " + name + " topics creation...")
        per_label_sentences = assignLabels(labels_per_scenario, sentences_per_scenario)
        topics_per_scenario = memoisation(outputPath + mode[0] + "/hdbscan", name + "_topics",
                                          lambda _ : createTopics(per_label_sentences, max_size=10))
        
        print("hdbscan - " + name + " evaluation...")
        for i in range(len(scenarios)):
            scenario_name, content = scenarios[i]
            _, texts = texts_per_scenario[i]
            _, topics = topics_per_scenario[i]

            if len(topics) == 0:
                continue

            texts_words = []
            for text in texts:
                words = []
                for sentence in text:
                    for word in sentence.split(" "):
                        words.append(word)
                texts_words.append(words)
            texts = texts_words

            dictionary = Dictionary(texts)
            corpus = [dictionary.doc2bow(text) for text in texts]

            for coherence in ["c_v", "u_mass"]:
                outputFile = outputPath + mode[0] + "/hdbscan/" + name + "_" + scenario_name + "_" + coherence + ".csv"
                coherence_func(outputFile, topics, corpus, texts, dictionary, coherence)