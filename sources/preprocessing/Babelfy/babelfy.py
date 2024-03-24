from os import listdir
from pybabelfy import *
from distribution import cutByPack

babelapi = Babelfy()

# Clé Babelfy
BABEL_KEY = "bc4877a4-d59c-4100-aaa1-3f55fada8f83"

#####
# Formatte une notion dans un texte au bon format dans un CSV.
#####
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

#####
# Traduit un texte en CSV-Babelfy
#####
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

#####
# Traduit un fichier en CSV-Babelfy
#####
def fileToCSV(inputFolder, outputFolder, document):
    f = open(inputFolder + "/" + document, "r")
    textToCSV(f.read(), outputFolder, document)
    f.close()

#####
# Traduit les fichiers d'un dossier en CSV-Babelfy
#####
def folderToCSV(inputFolder, outputFolder):
    for document in listdir(inputFolder):
        print(document)
        fileToCSV(inputFolder, outputFolder, document)

if __name__ == '__main__':
    folderToCSV("data/Raw", "data/Raw+Babelfy")
