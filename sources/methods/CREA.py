from os import listdir
from concepts import Definition, Context
from sklearn import preprocessing
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np
import pandas
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

def contextToMatrix(context):
    BnIds = context.extension([])
    Documents = context.intension([])

    AND_ID_DOC = [[0 for _ in range(len(Documents))] for _ in range(len(BnIds))]
    AND_ID_ID = [[0 for _ in range(len(BnIds))] for _ in range(len(BnIds))]
    AND_DOC_DOC = [[0 for _ in range(len(Documents))] for _ in range(len(Documents))]
    ID = [0 for _ in range(len(BnIds))]
    DOC = [0 for _ in range(len(Documents))]

    # Parcours du treillis
    for extent, intent in context.lattice:
        # OCCURENCE D'ID
        for bn in range(len(extent)):
            a = BnIds.index(extent[bn])
            ID[a] += 1
            # OCCURENCE D'ID & ID
            for bn2 in range(bn + 1, len(extent)):
                b = BnIds.index(extent[bn2])
                AND_ID_ID[a][b] += 1
                AND_ID_ID[b][a] += 1
            # OCCURENCE D'ID & DOCUMENT
            for doc in range(len(intent)):
                b = Documents.index(intent[doc])
                AND_ID_DOC[a][b] += 1

        # OCCURENCE DE DOCUMENT
        for doc in range(len(intent)):
            a = Documents.index(intent[doc])
            DOC[a] += 1
            # OCCURENCE DE DOCUMENT & DOCUMENT
            for doc2 in range(doc + 1, len(intent)):
                b = Documents.index(intent[doc2])
                AND_DOC_DOC[a][b] += 1
                AND_DOC_DOC[b][a] += 1

    ID_DOC  = [[AND_ID_DOC[i][j]  / (ID[i]  + DOC[j] - AND_ID_DOC[i][j] ) for j in range(len(Documents))] for i in range(len(BnIds))]
    ID_ID   = [[1. if i==j else AND_ID_ID[i][j]   / (ID[i]  + ID[j]  - AND_ID_ID[i][j]  ) for j in range(len(BnIds))]     for i in range(len(BnIds))]
    DOC_DOC = [[1. if i==j else AND_DOC_DOC[i][j] / (DOC[i] + DOC[j] - AND_DOC_DOC[i][j]) for j in range(len(Documents))] for i in range(len(Documents))]

    # MutualImpact, DocSimilarity, IdSimilarity
    return ID_DOC, ID_ID, DOC_DOC

def dissimilarityToClusters(IdDissimilarity, IdLabels, outputFolder):
    in_data = pandas.DataFrame(data=IdDissimilarity, index=IdLabels, columns=IdLabels)
    in_labels = np.array(in_data.columns.values.tolist())

    data_CR = preprocessing.scale(in_data)
    data_LinkMatrix = linkage(data_CR, method='ward', metric='euclidean')

    nb_clusters = 8

    groupes_cah = fcluster(data_LinkMatrix, t=nb_clusters, criterion='maxclust')

    cluster_list = []
    nb_clusters = 0
    for val in groupes_cah:
        if (nb_clusters < val):
            nb_clusters = val
    for _ in range(nb_clusters + 1):
        cluster_list.append([])

    for index, value in enumerate(groupes_cah):
        cluster_list[value].append(index)
    
    for cluster in range(1, len(cluster_list)):
        for elt in range(len(cluster_list[cluster])):
            index = cluster_list[cluster][elt]

    CSV_OUT = open(outputFolder + '/clusters.csv', "w")
    OFS = ";"
    for cluster in range(1, len(cluster_list)):
        line = str(cluster)
        for elt in range(len(cluster_list[cluster])):
            index = cluster_list[cluster][elt]
            out_label = in_labels[index]
            EachLabel = str.strip(out_label)
            line = line + OFS + EachLabel
        line = line + "\n"
        CSV_OUT.write(line)
    CSV_OUT.close()

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

    c = Context.fromstring(d.tostring())
    _, _, IdSimilarity = contextToMatrix(c)

    # CLUSTERS
    dissimilarityToClusters([[1 - IdSimilarity[i][j] for j in range(l_ids)] for i in range(l_ids)], bnIds, outputFolder)

    # 

###########################################################
####################### APPLICATION #######################
###########################################################
dataCouples = [
        ("input-data/Raw+Babelfy/linked", "output-data/Raw+Babelfy+CREA"),
        ("input-data/Raw+RNNTagger+Babelfy/linked/keepPunctuation", "output-data/Raw+RNNTagger+Babelfy+CREA/keepPunctuation"),
        ("input-data/Raw+RNNTagger+Babelfy/linked/punctuationClean", "output-data/Raw+RNNTagger+Babelfy+CREA/punctuationClean"),
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