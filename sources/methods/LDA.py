from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary

from gensim.models.ldamodel import LdaModel

from pprint import pprint

# Exemple de documents
text_corpus = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey",
]


# Prétraitement des données
# Pour cet exemple simple, nous allons simplement séparer chaque document en mots
# Vous pouvez utiliser des techniques plus avancées pour nettoyer et prétraiter vos données

# Création du dictionnaire
dictionary = corpora.Dictionary([doc.lower().split() for doc in documents])

# Création du corpus
corpus = [dictionary.doc2bow(doc.lower().split()) for doc in documents]

# Entraînement du modèle LDA
lda_model = LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)

# Affichage des sujets
pprint(lda_model.print_topics())
print("~~~~~~~~~~~~")
print(lda_model.print_topics())