# 1 - Raw
Documents initiaux, juste après OCRisation.
# 2 - Babelfy
Tokens représentant les notions de l'entrée via Babelfy, et traitements associés.
## 2.1 - prelinked
Tokens représentant les mots de l'entrée (uniquement appliquée dans le cas Raw+Babelfy).
## 2.2 - linked
Tokens représentant les notions de l'entrée via Babelfy, avec pour chaque notion

- L'identifiant (bn:id) de la notion dans le réseau sémantique Babelnet.
- L'index du premier caractère du contenu dans le texte.
- L'index du dernière caractère du contenu dans le texte.
- Le contenu dans le texte entre les deux index.
- Trois métriques caractérisant la tokenisation et utilisées par la méthode CREA
    - Le "score" (sans attribut).
    - Le "score global".
    - Le "score de consistance".
## 2.3 - equivalent
Identifiants (bn:id) des tokens représentant les notions de l'entrée.
# 3 - RNNTagger
Tokens représentant les mots de l'entrée via RNNTagger, et traitements associés.
## 3.1 - prelemmatized
Tokens représentant les mots de l'entrée via RNNTagger, avec pour chaque notion

- Le contenu dans le texte.
- ???
- Le résultat de la tokenisation.
## 3.2 - lemmatized
Résultats des tokens représentant les mots de l'entrée s'ils existent, et le contenu dans le texte d'entrée sinon.
## 3.3 - punctuationClean
Résultats post-traités des tokens représentant les mots de l'entrée, après

- Une mise en minuscule.
- Un retrait de la ponctuation, des chiffres et des caractères issus des erreurs de l'OCR (. , « » ? ! [ ] ( ) ; % ` @ : ’ ⇒ < > $ # & ' " • = ● \ / – 0 1 2 3 4 5 6 7 8 9 + - ◦ _ ˓ → … ~ { } * ^    © ° ´ |)
- Un retrait des mots inutiles (stopwords) de la langue française, tels que définis par nltk.
# 4 - TreeTagger
Tokens représentant les mots via TreeTagger, et traitements associés.
## 4.1 - prelemmatized
Tokens représentant les mots de l'entrée après

- Une mise en minuscule.
- Un retrait de la ponctuation, des chiffres et des caractères issus des erreurs de l'OCR (. , « » ? ! [ ] ( ) ; % ` @ : ’ ⇒ < > $ # & ' " • = ● \ / – 0 1 2 3 4 5 6 7 8 9 + - ◦ _ ˓ → … ~ { } * ^    © ° ´ |)
- Possiblement un retrait des mots inutiles (stopwords) de la langue française, tels que définis par nltk.
### 4.1.1 - keepStopWords
Version dans laquelle les mots inutiles sont gardés.
### 4.1.2 - throwStopWords
Version dans laquelle les mots inutiles sont supprimés.
## 4.2 - lemmatized
Tokens représentant les mots de l'entrée via TreeTagger. Les ambiguités (comme "cours|cour" pour le mot "cours", ou "le|la" pour le mot "les") sont résolues en gardant la première proposition.
### 4.2.1 - keepStopData
Version dans laquelle les mots inutiles sont gardés.
### 4.2.2 - throwStopClasses
Version dans laquelle les tokens inutiles sont supprimés si leur classe TreeTagger appartient à ['ADV', 'DET:ART', 'DET:POS', 'KON', 'PRO', 'PRO:DEM', 'PRO:IND', 'PRO:PER', 'PRO:POS', 'PRO:REL', 'PRP:det'].
### 4.2.3 - throwStopWords
Version dans laquelle les mots inutiles sont supprimés durant la phase de pré-lemmatisation.
