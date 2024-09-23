from os import listdir
from os.path import isdir
from ctc.main import Auto_CTC

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
def coherence_func(outputFilePrefix, topics, texts):
    ### CPMI Evaluation
    eval=Auto_CTC(segments_length=15, min_segment_length=5, segment_step=10,device="cpu") 
    eval.segmenting_documents(texts) 
    eval.create_cpmi_tree()

    cpmi = eval.ctc_cpmi(topics)

    f = open(outputFilePrefix + "cpmi.txt", "w")
    f.write("coherence;" + str(cpmi))
    f.close()

    ### Intrusion and rating Evaluation OF TOPICS ONLY (independant of texts)
    # openai_key=None
    eval=Semi_auto_CTC(None, topics)

    intrusion = eval.ctc_intrusion()
    f = open(outputFilePrefix + "intrusion.txt", "w")
    f.write("coherence;" + str(intrusion))
    f.close()

    rating = eval.ctc_rating()
    f = open(outputFilePrefix + "rating.txt", "w")
    f.write("coherence;" + str(intrusion))
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
        texts, docs = [], []
        def textToCorpus(text, document):
            texts.append(text)
            docs.append(document)
        folderToX("input-data/Raw", textToCorpus, scContent)

        # EVALUATIONS
        print(f"{scName} ({i+1}/{l_scs}) - Evaluations... (3/3)", flush=True)
        l_t = len(topicsDocs)
        for j in range(l_t):
            print(f"{scName} ({i+1}/{l_scs}) - Evaluations (3/3) - " + topicsDocs[j] + "_ctc_" + "{cpmi, intrusion, rating}" + f".txt... ({j+1}/{l_t})", flush=True)
            outputFilePrefix = "output/" + scName + "/evaluations/" + topicsDocs[j] + "_ctc_"
            coherence_func(outputFilePrefix, topicsCorpus[j], texts)