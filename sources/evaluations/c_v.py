from os import listdir
from os.path import isdir
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from preprocessing.tokenization import tokenize

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

def folderToX(inputFolder, textToX):
    for document in listdir(inputFolder):
        f = open(inputFolder + "/" + document, "r")
        textToX(f.read(), document)
        f.close()

##########################################################
###################### COHERENCE V #######################
##########################################################
def coherence_v(outputFile, topics, corpus, dictionary):
    cm = CoherenceModel(topics=topics, corpus=corpus, dictionary=dictionary, coherence='u_mass')
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
    # TOPICS LOADING
    print("Topics loading... (1/3)", flush=True)
    topicsCorpus, topicsDocs = [], []
    def textToCorpus(text, document):
        path,_ = document
        name = path.split("/")[1]

        inner_corpus = text.splitlines()
        for i in range(len(inner_corpus)):
            inner_corpus[i] = inner_corpus[i].split(";")[1:]

        topicsCorpus.append(inner_corpus)
        topicsDocs.append(name)
    folderToLoadWithX("output-data", textToCorpus)

    # RAW DOCUMENTS LOADING
    print("Raw documents loading... (2/3)", flush=True)
    corpus, docs = [], []
    def textToCorpus(text, document):
        corpus.append(text)
        docs.append(document)
    folderToX("input-data/Raw", textToCorpus)

    corpus = [tokenize(doc, True) for doc in corpus]
    dictionary = Dictionary(corpus)
    corpus = [dictionary.doc2bow(doc) for doc in corpus]

    # EVALUATIONS
    print("Evaluations... (3/3)", flush=True)
    for i in range(len(topicsDocs)):
        print("Evaluations (3/3) - " + topicsDocs[i] + ".txt...")
        outputFile = "output-evaluations/" + topicsDocs[i] + ".txt"
        coherence_v(outputFile, topicsCorpus[i], corpus, dictionary)