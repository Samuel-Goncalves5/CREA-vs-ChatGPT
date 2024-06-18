from os import listdir
from os.path import isdir
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from preprocessing.tokenization import tokenize

from generate_scenarios import getScenarios

##########################################################
################# RECHERCHE DES FICHIERS #################
##########################################################
def folderToLoadWithX(inputFolder, loadWithX):
    if "result_terms.txt" in listdir(inputFolder):
        path = inputFolder + "/result_terms.txt"
        document = (inputFolder, "result_terms.txt")
    elif "result.txt" in listdir(inputFolder):
        path = inputFolder + "/result.txt"
        document = (inputFolder, "result.txt")
    else:
        for document in listdir(inputFolder):
            path = inputFolder + "/" + document
            if isdir(path):
                folderToLoadWithX(path, loadWithX)
        return

    f = open(path, "r")
    loadWithX(f.read(), document)
    f.close()

def folderToX(inputFolder, textToX, docFilter):
    for document in listdir(inputFolder):
        if document.split(".")[0] in docFilter:
            f = open(inputFolder + "/" + document, "r")
            textToX(f.read(), document)
            f.close()

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

###########################################################
####################### APPLICATION #######################
###########################################################
if __name__ == '__main__':
    scs = getScenarios()
    l_scs = len(scs)

    for i in range(l_scs):
        scName, scContent = scs[i]

        # TOPICS LOADING
        print(f"{scName} ({i+1}/{l_scs}) - Topics loading... (1/3)", flush=True)
        topicsCorpus, topicsDocs = [], []
        def textToCorpus(text, document):
            path,_ = document
            name = path.split("/")[3]

            inner_corpus = text.splitlines()
            for i in range(len(inner_corpus)):
                inner_corpus[i] = inner_corpus[i].split(";")[1:]

            topicsCorpus.append(inner_corpus)
            topicsDocs.append(name)
        folderToLoadWithX("output/" + scName + "/data", textToCorpus)

        # RAW DOCUMENTS LOADING
        print(f"{scName} ({i+1}/{l_scs}) - Raw documents loading... (2/3)", flush=True)
        corpus, docs = [], []
        def textToCorpus(text, document):
            corpus.append(text)
            docs.append(document)
        folderToX("input-data/Raw", textToCorpus, scContent)

        corpus = [tokenize(doc, True) for doc in corpus]
        texts = corpus
        dictionary = Dictionary(corpus)
        corpus = [dictionary.doc2bow(doc) for doc in corpus]

        # EVALUATIONS
        print(f"{scName} ({i+1}/{l_scs}) - Evaluations... (3/3)", flush=True)
        l_t = len(topicsDocs)
        for j in range(l_t):
            for coherence in ['c_v', 'u_mass']:
                print(f"{scName} ({i+1}/{l_scs}) - Evaluations (3/3) - " + topicsDocs[j] + "_" + coherence + f".txt... ({j+1}/{l_t})", flush=True)
                outputFile = "output/" + scName + "/evaluations/" + topicsDocs[j] + "_" + coherence + ".txt"
                coherence_func(outputFile, topicsCorpus[j], corpus, texts, dictionary, coherence)