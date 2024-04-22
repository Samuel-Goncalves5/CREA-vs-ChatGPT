from os import listdir, system

RNNTagger = "preprocessing/RNNTagger/cmd/rnn-tagger-french.sh"
ExtractColumn = "preprocessing/extract_3rd_or_1st_column.awk"

##########################################################
############ APPLICATION A TOUS LES DOCUMENTS ############
##########################################################
def folderToX(inputFolder, outputFolder, fileToX):
    for document in listdir(inputFolder):
        print(document, flush=True)
        print(">", outputFolder + "/" + document)
        fileToX(inputFolder + "/" + document, outputFolder + "/" + document)

##########################################################
#################### PRELEMMATISATION ####################
##########################################################
def fileToPrelemmatised(inputDocument, outputDocument):
    system(f"{RNNTagger} {inputDocument} > {outputDocument}")

##########################################################
##################### LEMMATISATION ######################
##########################################################
def fileToLemmatised(inputDocument, outputDocument):
    system(f"awk -F '\t' -v OFS='\t' -f {ExtractColumn} '{inputDocument}' > '{outputDocument}'")

###########################################################
####################### APPLICATION #######################
###########################################################
if __name__ == '__main__':
    # TOKENISATION
    print("Prelemmatisation... (1/2)", flush=True)
    #folderToX("input-data/Raw", "input-data/Raw+RNNTagger/prelemmatized", fileToPrelemmatised)
    
    # LEMMATISATION
    print("Lemmatisation... (2/2)", flush=True)
    folderToX("input-data/Raw+RNNTagger/prelemmatized", "input-data/Raw+RNNTagger/lemmatized", fileToLemmatised)