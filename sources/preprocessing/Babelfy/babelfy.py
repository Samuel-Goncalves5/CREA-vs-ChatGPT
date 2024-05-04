from os import listdir
from tokenization import tokenize
from pybabelfy import Babelfy, AnnTypeValues
from distribution import cutByPack

# Clé Babelfy
BABEL_KEY = "bc4877a4-d59c-4100-aaa1-3f55fada8f83"

babelapi = Babelfy()

##########################################################
############ APPLICATION A TOUS LES DOCUMENTS ############
##########################################################
def folderToX(inputFolder, outputFolder, textToX):
    for document in listdir(inputFolder):
        print(document, flush=True)

        f = open(inputFolder + "/" + document, "r")
        textToX(f.read(), outputFolder, document)
        f.close()

##########################################################
###################### TOKENISATION ######################
##########################################################
def textToTokens(text, outputFolder, document):
    tokens = tokenize(text)

    token_database = open(outputFolder + "/" + document, "w")
    # Ajout des tokens au fichier
    for token in tokens:
        token_database.write(token + "\n")
    token_database.close()


##########################################################
##################### ENTITY LINKING #####################
##########################################################
def tokenToCSV(token, text, start_index):
    relative_start = token.char_fragment_start()
    relative_end   = token.char_fragment_end()

    bn_id  = str( token.babel_synset_id()      )
    start  = str( relative_start + start_index )
    end    = str( relative_end   + start_index )
    score  = str( token.score()                )
    gscore = str( token.global_score()         )
    cscore = str( token.coherence_score()      )

    return bn_id + ";" + start + ";" + end + ";" + " ".join(text[relative_start : relative_end + 1].split('\n')) + ";" + score + ";" + gscore + ";" + cscore + "\n"

def textToCSV(text, outputFolder, document):
    csv_database = open(outputFolder + "/" + document + ".csv", "w")
    # Couper le document en bloc de caractères
    requests, indexes = cutByPack(text)
    print([indexes[i] - indexes[i-1] for i in range(1, len(indexes))])
    # Parcours des blocs
    for i in range(len(requests)):
        # Babelisation
        try:
            tokens = babelapi.disambiguate(requests[i], lang="FR", key=BABEL_KEY, anntype=AnnTypeValues.ALL)
            # Ajout des tokens au fichier CSV
            for token in tokens:
                csv_database.write( tokenToCSV(token, requests[i], indexes[i]) )
        except Exception as e:
            print(repr(e))
    csv_database.close()

###########################################################
##################### EQUIVALENT TEXT #####################
###########################################################
def textToEquivalent(text, outputFolder, document):
    equivalent_database = open(outputFolder + "/" + document, "w")
    for line in text.splitlines():
        equivalent_database.write(line.rstrip().split(';')[0] + "\n")
    equivalent_database.close()

###########################################################
####################### APPLICATION #######################
###########################################################
if __name__ == '__main__':
    import sys

    if sys.argv[1] == "Raw":
        # TOKENISATION
        print("Tokenisation... (1/3)", flush=True)
        folderToX("input-data/Raw", "input-data/Raw+Babelfy/prelinked", textToTokens)
        # ENTITY LINKING
        print("Entity linking... (2/3)", flush=True)
        folderToX("input-data/Raw+Babelfy/prelinked", "input-data/Raw+Babelfy/linked", textToCSV)
        # EQUIVALENT TEXT
        print("Equivalent text... (3/3)", flush=True)
        folderToX("input-data/Raw+Babelfy/linked", "input-data/Raw+Babelfy/equivalent", textToEquivalent)
    
    elif sys.argv[1] == "RNNTagger":
        # ENTITY LINKING
        print("Entity linking... (1/2)", flush=True)
        folderToX("input-data/Raw+RNNTagger/lemmatized", "input-data/Raw+RNNTagger+Babelfy/linked", textToCSV)
        # EQUIVALENT TEXT
        print("Equivalent text... (2/2)", flush=True)
        folderToX("input-data/Raw+RNNTagger+Babelfy/linked", "input-data/Raw+RNNTagger+Babelfy/equivalent", textToEquivalent)
    
    elif sys.argv[1] == "TreeTagger-no-stop":
        # ENTITY LINKING
        print("Entity linking... (1/2)", flush=True)
        folderToX("input-data/Raw+TreeTagger/lemmatized/keepStopData", "input-data/Raw+TreeTagger+Babelfy/linked/keepStopData", textToCSV)
        # EQUIVALENT TEXT
        print("Equivalent text... (2/2)", flush=True)
        folderToX("input-data/Raw+TreeTagger+Babelfy/linked/keepStopData", "input-data/Raw+TreeTagger+Babelfy/equivalent/keepStopData", textToEquivalent)
    
    elif sys.argv[1] == "TreeTagger-stop-class":
        # ENTITY LINKING
        print("Entity linking... (1/2)", flush=True)
        folderToX("input-data/Raw+TreeTagger/lemmatized/throwStopClasses", "input-data/Raw+TreeTagger+Babelfy/linked/throwStopClasses", textToCSV)
        # EQUIVALENT TEXT
        print("Equivalent text... (2/2)", flush=True)
        folderToX("input-data/Raw+TreeTagger+Babelfy/linked/throwStopClasses", "input-data/Raw+TreeTagger+Babelfy/equivalent/throwStopClasses", textToEquivalent)
    
    elif sys.argv[1] == "TreeTagger-stop-word":
        # ENTITY LINKING
        print("Entity linking... (1/2)", flush=True)
        folderToX("input-data/Raw+TreeTagger/lemmatized/throwStopWords", "input-data/Raw+TreeTagger+Babelfy/linked/throwStopWords", textToCSV)
        # EQUIVALENT TEXT
        print("Equivalent text... (2/2)", flush=True)
        folderToX("input-data/Raw+TreeTagger+Babelfy/linked/throwStopWords", "input-data/Raw+TreeTagger+Babelfy/equivalent/throwStopWords", textToEquivalent)

    else:
        print("Error : Valid arguments are : Raw - RNNTagger - TreeTagger-no-stop - TreeTagger-stop-class - TreeTagger-stop-word")