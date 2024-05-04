from os import listdir
from concepts import Definition, Context
import json

##########################################################
############ APPLICATION A TOUS LES DOCUMENTS ############
##########################################################
def folderToX(inputFolder, fileToX):
    for document in listdir(inputFolder):
        fileToX(inputFolder + "/" + document, document)

##########################################################
########################## CREA ##########################
##########################################################
FILTER_LIMIT = 0.5
def crea(outputFolder, BabelDictionaries, bnIds, docs):
    l_ids = len(bnIds)
    l_docs = len(docs)

    # OCCURENCES
    def occurencesFor(i,j):
        if bnIds[j] in BabelDictionaries[i]:
            return len(BabelDictionaries[i][bnIds[j]])
        else:
            return 0
    Entity_Matrix = [[occurencesFor(i,j) for j in range(l_ids)] for i in range(l_docs)]

    # DOUBLE NORMALISATION
    for occurences in Entity_Matrix:
        s = sum(occurences)
        for i in range(l_ids):
            occurences[i] /= s
    Entity_Matrix = [[Entity_Matrix[j][i] for j in range(l_docs)] for i in range(l_ids)]
    for occurences in Entity_Matrix:
        s = sum(occurences)
        for i in range(l_docs):
            occurences[i] /= s

    # FORMAL CONCEPT ANALYSIS
    d = Definition()
    for i in range(l_ids):
        documents = []
        for j in range(l_docs):
            if Entity_Matrix[i][j] > 0.0:
                documents.append(docs[j])
        d.add_property(bnIds[i], documents)
    print(d)

###########################################################
####################### APPLICATION #######################
###########################################################
dataCouples = [
        ("input-data/Raw+Babelfy/linked", "output-data/Raw+Babelfy+CREA"),
        ("input-data/Raw+RNNTagger+Babelfy/linked", "output-data/Raw+RNNTagger+Babelfy+CREA"),
        ("input-data/Raw+TreeTagger+Babelfy/linked/keepStopData", "output-data/Raw+TreeTagger+Babelfy+CREA/keepStopData"),
        ("input-data/Raw+TreeTagger+Babelfy/linked/throwStopClasses", "output-data/Raw+TreeTagger+Babelfy+CREA/throwStopClasses"),
        ("input-data/Raw+TreeTagger+Babelfy/linked/throwStopWords", "output-data/Raw+TreeTagger+Babelfy+CREA/throwStopWords"),
    ]

if __name__ == '__main__':
    # CREA
    print("CREA...", flush=True)
    l_data = len(dataCouples)
    for i in range(l_data):
        inputFolder, outputFolder = dataCouples[i]
        print(f"CREA - {inputFolder}... ({i+1}/{l_data})", flush=True)
        bnIds, docs = [], []
        BabelDictionaries = []
        def fileToDictionary(f, document):
            docs.append(document)
            documentBabelDictionary = {}
            with open(f, "r") as file:
                for line in file:
                    values = line.rstrip().split(';')
                    if float(values[-1]) >= FILTER_LIMIT:
                        if not values[0] in bnIds:
                            bnIds.append(values[0])
                        
                        if values[0] in documentBabelDictionary:
                            documentBabelDictionary[values[0]].append(values)
                        else:
                            documentBabelDictionary[values[0]] = [values]

            BabelDictionaries.append(documentBabelDictionary)

        folderToX(inputFolder, fileToDictionary)
        crea(outputFolder, BabelDictionaries, bnIds, docs)