from os import listdir
from gensim.models.ldamodel import LdaModel
from gensim.matutils import corpus2csc
from gensim.corpora import Dictionary
import json

##########################################################
############ APPLICATION A TOUS LES DOCUMENTS ############
##########################################################
def folderToX(inputFolder, textToX):
    for document in listdir(inputFolder):
        f = open(inputFolder + "/" + document, "r")
        textToX(f.read(), document)
        f.close()

##########################################################
########################## LDA ###########################
##########################################################
TOPIC_NUMBER = 2
TOPIC_SIZE = 3
def lda(outputFolder, corpus, docs):
    dictionary = Dictionary(corpus)
    bagOfWords = [dictionary.doc2bow(doc) for doc in corpus]
    lda_model = LdaModel(corpus=bagOfWords, id2word=dictionary, num_topics=TOPIC_NUMBER)

    topics_json = []
    for i in range(TOPIC_NUMBER):
        topic = lda_model.show_topic(topicid=i, topn=TOPIC_SIZE)
        topic_json = []
        for word, score in topic:
            word_json = {
                "word": word,
                "score": float(score)
            }
            topic_json.append(word_json)

        topics_json.append(topic_json)

    documentsTopics = corpus2csc(lda_model.get_document_topics(bagOfWords)).T.toarray().tolist()
    documents_json = []
    for i in range(len(docs)):
        document_json = {
            "document": docs[i],
            "topics": documentsTopics[i]
        }
        documents_json.append(document_json)

    result_json = {
        "topics": topics_json,
        "documents": documents_json
    }

    with open(outputFolder + "/result", "w") as outputFile:
        json.dump(result_json, outputFile, indent=2)

###########################################################
####################### APPLICATION #######################
###########################################################
dataCouples = [
        ("input-data/Raw+Babelfy/prelinked", "output-data/Raw+LDA"),
        ("input-data/Raw+Babelfy/equivalent", "output-data/Raw+Babelfy+LDA"),
        ("input-data/Raw+RNNTagger/punctuationClean", "output-data/Raw+RNNTagger+LDA"),
        ("input-data/Raw+RNNTagger+Babelfy/equivalent", "output-data/Raw+RNNTagger+Babelfy+LDA"),
        ("input-data/Raw+TreeTagger/lemmatized/keepStopData", "output-data/Raw+TreeTagger+LDA/keepStopData"),
        ("input-data/Raw+TreeTagger/lemmatized/throwStopClasses", "output-data/Raw+TreeTagger+LDA/throwStopClasses"),
        ("input-data/Raw+TreeTagger/lemmatized/throwStopWords", "output-data/Raw+TreeTagger+LDA/throwStopWords"),
        ("input-data/Raw+TreeTagger+Babelfy/equivalent/keepStopData", "output-data/Raw+TreeTagger+Babelfy+LDA/keepStopData"),
        ("input-data/Raw+TreeTagger+Babelfy/equivalent/throwStopClasses", "output-data/Raw+TreeTagger+Babelfy+LDA/throwStopClasses"),
        ("input-data/Raw+TreeTagger+Babelfy/equivalent/throwStopWords", "output-data/Raw+TreeTagger+Babelfy+LDA/throwStopWords"),
    ]

if __name__ == '__main__':
    # LDA
    print("LDA...", flush=True)
    l_data = len(dataCouples)
    for i in range(l_data):
        inputFolder, outputFolder = dataCouples[i]
        print(f"LDA - {inputFolder}... ({i+1}/{l_data})", flush=True)
        corpus, docs = [], []
        def textToCorpus(text, document):
            corpus.append(text.splitlines())
            docs.append(document)
        
        folderToX(inputFolder, textToCorpus)
        lda(outputFolder, corpus, docs)