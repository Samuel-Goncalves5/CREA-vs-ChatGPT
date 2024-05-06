from os import listdir
from tokenization import tokenize
import treetaggerwrapper

TreeTagger = "preprocessing/TreeTagger"
tagger = treetaggerwrapper.TreeTagger(TAGLANG='fr', TAGDIR=TreeTagger)

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
throwStopWords = False
def textToTokens(text, outputFolder, document):
    tokens = tokenize(text, throwStopWords)

    token_database = open(outputFolder + "/" + document, "w")
    # Ajout des tokens au fichier
    for token in tokens:
        token_database.write(token + "\n")
    token_database.close()

##########################################################
##################### LEMMATISATION ######################
##########################################################
throwStopWords = False
classFilter = ['ADV', 'DET:ART', 'DET:POS', 'KON', 'PRO', 'PRO:DEM', 'PRO:IND', 'PRO:PER', 'PRO:POS', 'PRO:REL', 'PRP:det']
def textToTreeTagger(text, outputFolder, document):
    lemmes_database = open(outputFolder + "/" + document, "w")
    lemmes = treetaggerwrapper.make_tags(tagger.tag_text(text), allow_extra=True)
    # Ajout des lemmes au fichier
    for lemme in lemmes:
        if hasattr(lemme, "what"):
            continue
        if lemme.lemma == "@card@":
            continue
        if throwStopWords and lemme.pos in classFilter:
            continue

        # Remove disambiguations (mois|moi / être|suivre)
        lemmes_database.write(lemme.lemma.split('|')[0] + "\n")
    lemmes_database.close()
###########################################################
####################### APPLICATION #######################
###########################################################
if __name__ == '__main__':
    # TOKENISATION
    print("Tokenisation... (1/2)", flush=True)

    throwStopWords = True
    print("Tokenisation - throwStopWords... (1/2) (1/2)", flush=True)
    folderToX("input-data/Raw", "input-data/Raw+TreeTagger/prelemmatized/throwStopWords", textToTokens)
    
    throwStopWords = False
    print("Tokenisation - keepStopWords... (1/2) (2/2)", flush=True)
    folderToX("input-data/Raw", "input-data/Raw+TreeTagger/prelemmatized/keepStopWords", textToTokens)
    
    # LEMMATISATION
    print("Lemmatisation... (2/2)", flush=True)
    
    throwStopWords = True
    print("Lemmatisation - throwStopClasses... (2/2) (1/3)", flush=True)
    folderToX("input-data/Raw+TreeTagger/prelemmatized/keepStopWords", "input-data/Raw+TreeTagger/lemmatized/throwStopClasses", textToTreeTagger)
    
    throwStopWords = False
    print("Lemmatisation - throwStopWords... (2/2) (2/3)", flush=True)
    folderToX("input-data/Raw+TreeTagger/prelemmatized/throwStopWords", "input-data/Raw+TreeTagger/lemmatized/throwStopWords", textToTreeTagger)
    print("Lemmatisation - keepStopWords... (2/2) (3/3)", flush=True)
    folderToX("input-data/Raw+TreeTagger/prelemmatized/keepStopWords", "input-data/Raw+TreeTagger/lemmatized/keepStopData", textToTreeTagger)