# Comparaison de méthodes de modélisation de sujets face à ChatGPT et aux LLM
## I - Description
ChatGPT et les LLM ont radicalement changé notre relation aux connaissances et à l'interrogation des bases de connaissances : on ne demande plus à Google d'afficher le résumé Wikipedia d'un mot clé, mais on pose directement la question à ChatGPT (ou un assistant similaire). D'autres usages offrent autant d'opportunités : on peut demander à ChatGPT de résumer des textes. La recherche visée consiste à établir un état de l'art sur les LLM et ChatGPT, à développer un protocole expérimental et à l'exécuter pour comparer la qualité des résultats de ChatGPT et des LLM avec des méthodes plus classiques de modélisation de sujets et d'extraction de connaissances comme la méthode CREA [1] et l'Allocation de Dirichlet latente [2].

*[1]  F. Boissier, “CREA : méthode d’analyse, d’adaptation et de réutilisation des processus à forte intensité de connaissance - cas d’utilisation dans l’enseignement supérieur en informatique,” Ph.D. dissertation, Paris 1, 2022.*

*[2] D. M. Blei, A. Y. Ng, and M. I. Jordan, “Latent dirichlet allocation,” Journal of machine Learning research, vol. 3, no. Jan, pp. 993–1022, 2003.*

## II - Utilisation
### 1 - Outils
- **make help** : affiche le contenu de `README.md`
- **make build** : (Lancé automatiquement par les commandes suivantes) créé l'environnement (image Docker) d'utilisation des codes
- **make print-files** : affiche le contenu de l'environnement d'utilisation des codes (le chemin de chaque fichier)
### 2 - Protocole
#### 2.1 - Pré-traitement
##### 2.1.1 - RNNTagger
- **make preprocessing-rnn** :
    - génère la pré-lemmatisation RNNTagger des fichiers de `Raw/` dans `Raw+RNNTagger/prelemmatized/`
    - génère la lemmatisation des fichiers de `Raw+RNNTagger/prelemmatized/` dans `Raw+TreeTagger/lemmatized/`
    - génère la version sans ponctuation des fichiers de `Raw+TreeTagger/lemmatized/` dans `Raw+TreeTagger/punctuationClean/`
##### 2.1.2 - TreeTagger
- **make preprocessing-tree** :
    - génère la séparation en tokens des fichiers de `Raw/` dans `Raw+TreeTagger/prelemmatized/throwStopWords/` en filtrant les stop words et dans `Raw+TreeTagger/prelemmatized/keepStopWords/` en ne les filtrant pas
    - génère la lemmatisation TreeTagger des fichiers de `Raw+TreeTagger/prelemmatized/throwStopWords/` dans `Raw+TreeTagger/lemmatized/throwStopWords/`
    - génère la lemmatisation TreeTagger des fichiers de `Raw+TreeTagger/prelemmatized/keepStopWords/` dans `Raw+TreeTagger/lemmatized/throwStopClasses/` en filtrant les stop classes et dans `Raw+TreeTagger/lemmatized/keepStopData/` en ne les filtrant pas.
##### 2.1.3 - Babelfy
- **make preprocessing-babelfy-raw** :
    - génère la séparation en tokens des fichiers de `Raw/` dans `Raw+Babelfy/prelinked/`
    - génère les informations babelfy des fichiers de `Raw+Babelfy/prelinked/` dans `Raw+Babelfy/linked/`
    - génère les textes équivalents des fichiers de `Raw+Babelfy/linked/` dans `Raw+Babelfy/equivalent/`
- **make preprocessing-babelfy-rnn-punctuation** :
    - génère les informations babelfy des fichiers de `Raw+RNNTagger/lemmatized/` dans `Raw+RNNTagger+Babelfy/linked/keepPunctuation/`
    - génère les textes équivalents des fichiers de `Raw+RNNTagger+Babelfy/linked/keepPunctuation/` dans `Raw+RNNTagger+Babelfy/equivalent/keepPunctuation/`
- **make preprocessing-babelfy-rnn-no-punctuation** :
    - génère les informations babelfy des fichiers de `Raw+RNNTagger/punctuationClean/` dans `Raw+RNNTagger+Babelfy/linked/punctuationClean/`
    - génère les textes équivalents des fichiers de `Raw+RNNTagger+Babelfy/linked/punctuationClean/` dans `Raw+RNNTagger+Babelfy/equivalent/punctuationClean/`
- **make preprocessing-babelfy-tree-no-stop** :
    - génère les informations babelfy des fichiers de `Raw+TreeTagger/lemmatized/keepStopData/` dans `Raw+TreeTagger+Babelfy/linked/keepStopData/`
    - génère les textes équivalents des fichiers de `Raw+TreeTagger+Babelfy/linked/keepStopData/` dans `Raw+TreeTagger+Babelfy/equivalent/keepStopData/`
- **make preprocessing-babelfy-tree-stop-word** :
    - génère les informations babelfy des fichiers de `Raw+TreeTagger/lemmatized/throwStopWords/` dans `Raw+TreeTagger+Babelfy/linked/throwStopWords/`
    - génère les textes équivalents des fichiers de `Raw+TreeTagger+Babelfy/linked/throwStopWords/` dans `Raw+TreeTagger+Babelfy/equivalent/throwStopWords/`
- **make preprocessing-babelfy-tree-stop-class** :
    - génère les informations babelfy des fichiers de `Raw+TreeTagger/lemmatized/throwStopClasses/` dans `Raw+TreeTagger+Babelfy/linked/throwStopClasses/`
    - génère les textes équivalents des fichiers de `Raw+TreeTagger+Babelfy/linked/throwStopClasses/` dans `Raw+TreeTagger+Babelfy/equivalent/throwStopClasses/`
#### 2.2 - Traitement
##### 2.2.1 - CREA
- **make crea** :
    génère pour chacun des dossiers suivants la modélisation des sujets et le niveau d'appartenance de chaque document du dossier à chaque classe selon la méthode CREA:
    - De `input/data/Raw+Babelfy/linked/` à `output/data/Raw+Babelfy+CREA`
    - De `input/data/Raw+RNNTagger+Babelfy/linked/keepPunctuation/` à `output/data/Raw+RNNTagger+Babelfy+CREA/keepPunctuation/`
    - De `input/data/Raw+RNNTagger+Babelfy/linked/punctuationClean/` à `output/data/Raw+RNNTagger+Babelfy+CREA/punctuationClean/`
    - De `input/data/Raw+TreeTagger+Babelfy/linked/keepStopData/` à `output-data/Raw+TreeTagger+Babelfy+CREA/keepStopData/`
    - De `input/data/Raw+TreeTagger+Babelfy/linked/throwStopClasses/` à `output-data/Raw+TreeTagger+Babelfy+CREA/throwStopClasses/`
    - De `input/data/Raw+TreeTagger+Babelfy/linked/throwStopWords/` à `output-data/Raw+TreeTagger+Babelfy+CREA/throwStopWords/`
##### 2.2.2 - GPT
TODO
##### 2.2.3 - LDA
- **make lda** :
    génère pour chacun des dossiers suivants la modélisation des sujets et le niveau d'appartenance de chaque document du dossier à chaque classe selon la méthode LDA:
    - De `input/data/Raw+Babelfy/prelinked/` à `output/data/Raw+LDA/`
    - De `input/data/Raw+Babelfy/equivalent/` à `output/data/Raw+Babelfy+LDA/`
    - De `input/data/Raw+RNNTagger/punctuationClean/` à `output/data/Raw+RNNTagger+LDA/`
    - De `input/data/Raw+RNNTagger+Babelfy/equivalent/punctuationClean/` à `output/data/Raw+RNNTagger+Babelfy+LDA/punctuationClean/`
    - De `input/data/Raw+RNNTagger+Babelfy/equivalent/keepPunctuation/` à `output/data/Raw+RNNTagger+Babelfy+LDA/keepPunctuation/`
    - De `input/data/Raw+TreeTagger/lemmatized/keepStopData/` à `output/data/Raw+TreeTagger+LDA/keepStopData/`
    - De `input/data/Raw+TreeTagger/lemmatized/throwStopClasses/` à `output/data/Raw+TreeTagger+LDA/throwStopClasses/`
    - De `input/data/Raw+TreeTagger/lemmatized/throwStopWords/` à `output/data/Raw+TreeTagger+LDA/throwStopWords/`
    - De `input/data/Raw+TreeTagger+Babelfy/equivalent/keepStopData/` à `output/data/Raw+TreeTagger+Babelfy+LDA/keepStopData/`
    - De `input/data/Raw+TreeTagger+Babelfy/equivalent/throwStopClasses/` à `output/data/Raw+TreeTagger+Babelfy+LDA/throwStopClasses/`
    - De `input/data/Raw+TreeTagger+Babelfy/equivalent/throwStopWords/` à `output/data/Raw+TreeTagger+Babelfy+LDA/throwStopWords/`
##### 2.2.4 - Llama2
TODO
#### 2.3 - Evaluation
##### 2.3.1 - Cohérence V
TODO