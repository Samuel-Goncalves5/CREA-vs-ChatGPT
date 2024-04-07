# Comparaison de méthodes de modélisation de sujets face à ChatGPT et aux LLM
## Description
ChatGPT et les LLM ont radicalement changé notre relation aux connaissances et à l'interrogation des bases de connaissances : on ne demande plus à Google d'afficher le résumé Wikipedia d'un mot clé, mais on pose directement la question à ChatGPT (ou un assistant similaire). D'autres usages offrent autant d'opportunités : on peut demander à ChatGPT de résumer des textes. La recherche visée consiste à établir un état de l'art sur les LLM et ChatGPT, à développer un protocole expérimental et à l'exécuter pour comparer la qualité des résultats de ChatGPT et des LLM avec des méthodes plus classiques de modélisation de sujets et d'extraction de connaissances comme la méthode CREA [1] et l'Allocation de Dirichlet latente [2].

*[1]  F. Boissier, “CREA : méthode d’analyse, d’adaptation et de réutilisation des processus à forte intensité de connaissance - cas d’utilisation dans l’enseignement supérieur en informatique,” Ph.D. dissertation, Paris 1, 2022.*

*[2] D. M. Blei, A. Y. Ng, and M. I. Jordan, “Latent dirichlet allocation,” Journal of machine Learning research, vol. 3, no. Jan, pp. 993–1022, 2003.*

## Utilisation
### Outils
- make help : affiche le contenu de `README.md`
- make build : (Lancé automatiquement par les commandes suivantes) créé l'environnement (image Docker) d'utilisation des codes
- make print-files : affiche le contenu de l'environnement d'utilisation des codes (le chemin de chaque fichier)
### Protocole
#### Pré-traitement
##### RNNTagger
Fichiers et codes préalablement fournis.
##### TreeTagger
TODO
##### Babelfy
- make preprocessing-babelfy-raw :
    - génère la séparation en tokens des fichiers de Raw dans Raw+Babelfy/prelinked
    - génère les informations babelfy des fichiers de Raw+Babelfy/prelinked dans Raw+Babelfy/linked
    - génère les textes équivalents des fichiers de Raw+Babelfy/linked dans Raw+Babelfy/equivalent
- make preprocessing-babelfy-rnn :
    - génère les informations babelfy des fichiers de Raw+RNNTagger/lemmatized dans Raw+RNNTagger+Babelfy/linked
    - génère les textes équivalents des fichiers de Raw+RNNTagger+Babelfy/linked dans Raw+RNNTagger+Babelfy/equivalent
- make preprocessing-babelfy-tree :
    - TODO
#### Traitement
##### CREA
TODO
##### GPT
TODO
##### LDA
TODO
##### Llama2
TODO
#### Evaluation
##### Cohérence V
TODO

## Méthodes
- `ChatGPT.py` permet d'utiliser l'API de ChatGPT.
- `Llama2.py` permet d'utiliser l'API de Llama 2.
- `CREA.py` permet d'utiliser la méthode CREA.
- `LDA.py` permet d'utiliser la LDA.
