from nltk.corpus import stopwords
from nltk import word_tokenize
import re

frUseless = set(stopwords.words('french'))
frOther = ["a", "le", "les", "des", "un", "une", "où", "ou",
            "l", "ni", "si", "ce", "cette", "cet", "donc", "dont"]
frUseless.update(frOther)

externFiltre = lambda text: [token.lower() for token in text if token.lower() not in frUseless]

ponctuation = ['.', ',', '«', '»', '?', '!', '[', ']', '(', ')', ';', '%', '`', '@', ':', '’',
               '⇒', '<', '>', '\'', '$', '#', '&', "'", '"', '•', '=', '●', '\\', '/', '–',
               '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '◦', '_', '˓', '→',
               '|', '…', '~', '{', '}', '*', '^', '', '', '', '©', '°', '´']
frEliminateDico = str.maketrans('','', "".join(ponctuation))
internFiltre = lambda text: [token.translate(frEliminateDico) for token in text if len(token.translate(frEliminateDico)) > 0]

def tokenize(text, throwStopWords=False):
    ponctuationRegex = '|'.join(map(re.escape, ponctuation))
    text = re.sub(f'({ponctuationRegex})', r' \1 ', text)
    tokens = word_tokenize(text, language="french")

    tokens = internFiltre(tokens)
    if not throwStopWords:
        tokens = externFiltre(tokens)

    return tokens