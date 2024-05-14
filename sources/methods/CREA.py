from os import listdir
from concepts import Definition, Context
from sklearn import preprocessing
from scipy.cluster.hierarchy import linkage, fcluster
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
TOPIC_NUMBER = 8

def contextToMatrix(context):
    Documents = context.extension([])
    BnIds = context.intension([])

    AND_ID_DOC =  [[0 for _ in range(len(Documents))] for _ in range(len(BnIds    ))]
    AND_ID_ID =   [[0 for _ in range(len(BnIds    ))] for _ in range(len(BnIds    ))]
    AND_DOC_DOC = [[0 for _ in range(len(Documents))] for _ in range(len(Documents))]
    ID  = [0 for _ in range(len(BnIds    ))]
    DOC = [0 for _ in range(len(Documents))]

    # Parcours du treillis
    for extent, intent in context.lattice:
        # OCCURENCE D'ID
        for bn in range(len(intent)):
            a = BnIds.index(intent[bn])
            ID[a] += 1
            # OCCURENCE D'ID & ID
            for bn2 in range(bn + 1, len(intent)):
                b = BnIds.index(intent[bn2])
                AND_ID_ID[a][b] += 1
                AND_ID_ID[b][a] += 1
            # OCCURENCE D'ID & DOCUMENT
            for doc in range(len(extent)):
                b = Documents.index(extent[doc])
                AND_ID_DOC[a][b] += 1

        # OCCURENCE DE DOCUMENT
        for doc in range(len(extent)):
            a = Documents.index(extent[doc])
            DOC[a] += 1
            # OCCURENCE DE DOCUMENT & DOCUMENT
            for doc2 in range(doc + 1, len(extent)):
                b = Documents.index(extent[doc2])
                AND_DOC_DOC[a][b] += 1
                AND_DOC_DOC[b][a] += 1

    ID_DOC  = [[                AND_ID_DOC[i][j]  / (ID[i]  + DOC[j] - AND_ID_DOC[i][j] ) for j in range(len(Documents))] for i in range(len(BnIds    ))]
    ID_ID   = [[1. if i==j else AND_ID_ID[i][j]   / (ID[i]  + ID[j]  - AND_ID_ID[i][j]  ) for j in range(len(BnIds    ))] for i in range(len(BnIds    ))]
    DOC_DOC = [[1. if i==j else AND_DOC_DOC[i][j] / (DOC[i] + DOC[j] - AND_DOC_DOC[i][j]) for j in range(len(Documents))] for i in range(len(Documents))]

    # MutualImpact, DocSimilarity, IdSimilarity
    return ID_DOC, DOC_DOC, ID_ID

def clusterizedContextToMatrix(context, cluster_tags, cluster_nb):
    Documents = context.extension([])
    BnIds = context.intension([])

    AND_CLUSTER_DOC     = [[0 for _ in range(len(Documents))] for _ in range(cluster_nb    )]
    AND_CLUSTER_CLUSTER = [[0 for _ in range(cluster_nb    )] for _ in range(cluster_nb    )]
    AND_DOC_DOC         = [[0 for _ in range(len(Documents))] for _ in range(len(Documents))]
    CLUSTER = [0 for _ in range(cluster_nb    )]
    DOC     = [0 for _ in range(len(Documents))]

    # Parcours du treillis
    for extent, intent in context.lattice:
        nodeClusters = []
        for bn in range(len(intent)):
            # OCCURENCE DE CLUSTER
            clust = cluster_tags[BnIds.index(intent[bn])]
            if not clust in nodeClusters:
                nodeClusters.append(clust)
                CLUSTER[clust] += 1

        for i in range(len(nodeClusters)):
            # OCCURENCE DE CLUSTER & CLUSTER
            for j in range(i+1, len(nodeClusters)):
                AND_CLUSTER_CLUSTER[nodeClusters[i]][nodeClusters[j]] += 1
                AND_CLUSTER_CLUSTER[nodeClusters[j]][nodeClusters[i]] += 1
            # OCCURENCE DE CLUSTER & DOCUMENT
            for doc in range(len(extent)):
                b = Documents.index(extent[doc])
                AND_CLUSTER_DOC[nodeClusters[i]][b] += 1

        # OCCURENCE DE DOCUMENT
        for doc in range(len(extent)):
            a = Documents.index(extent[doc])
            DOC[a] += 1
            # OCCURENCE DE DOCUMENT & DOCUMENT
            for doc2 in range(doc + 1, len(extent)):
                b = Documents.index(extent[doc2])
                AND_DOC_DOC[a][b] += 1
                AND_DOC_DOC[b][a] += 1

    CLUSTER_DOC     = [[                AND_CLUSTER_DOC[i][j]     / (CLUSTER[i]  + DOC[j]     - AND_CLUSTER_DOC[i][j]    ) for j in range(len(Documents))] for i in range(cluster_nb    )]
    CLUSTER_CLUSTER = [[1. if i==j else AND_CLUSTER_CLUSTER[i][j] / (CLUSTER[i]  + CLUSTER[j] - AND_CLUSTER_CLUSTER[i][j]) for j in range(cluster_nb    )] for i in range(cluster_nb    )]
    DOC_DOC         = [[1. if i==j else AND_DOC_DOC[i][j]         / (DOC[i]      + DOC[j]     - AND_DOC_DOC[i][j]        ) for j in range(len(Documents))] for i in range(len(Documents))]

    # MutualImpact, DocSimilarity, IdSimilarity
    return CLUSTER_DOC, DOC_DOC, CLUSTER_CLUSTER

def dissimilarityToClusters(IdDissimilarity, IdLabels):
    in_data = pandas.DataFrame(data=IdDissimilarity, index=IdLabels, columns=IdLabels)
    data_CR = preprocessing.scale(in_data)
    data_LinkMatrix = linkage(data_CR, method='ward', metric='euclidean')

    groupes_cah = fcluster(data_LinkMatrix, t=TOPIC_NUMBER, criterion='maxclust')
    cluster_list = [[] for _ in range(max(groupes_cah))]

    for i in range(len(groupes_cah)):
        groupes_cah[i] -= 1

    for index, cluster in enumerate(groupes_cah):
        cluster_list[cluster].append(index)

    return groupes_cah, cluster_list

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
    cluster_tags, clusters = dissimilarityToClusters([[1 - IdSimilarity[i][j] for j in range(l_ids)] for i in range(l_ids)], bnIds)

    # CLUSTERIZED FORMAL CONCEPT ANALYSIS
    clusterizedMutualImpact, _, _ = clusterizedContextToMatrix(c, cluster_tags, len(clusters))

    # NORMALISATION
    clusterizedMutualImpact = [[clusterizedMutualImpact[i][j] for i in range(len(clusterizedMutualImpact))] for j in range(len(clusterizedMutualImpact[0]))]
    for clusterImpactByDocument in clusterizedMutualImpact:
        s = sum(clusterImpactByDocument)
        for clusterImpactIndex in range(len(clusterImpactByDocument)):
            clusterImpactByDocument[clusterImpactIndex] /= s

    # JSON
    topics_json = []
    for cluster in clusters:
        topic_json = []
        for index in cluster:
            word_json = {
                "word": bnIds[index]
            }
            topic_json.append(word_json)

        topics_json.append(topic_json)

    documents_json = []
    for i in range(len(docs)):
        document_json = {
            "document": docs[i],
            "topics": clusterizedMutualImpact[i]
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