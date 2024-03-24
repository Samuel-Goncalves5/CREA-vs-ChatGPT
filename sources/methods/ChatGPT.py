import os
from openai import OpenAI

# clé API à générer depuis : https://platform.openai.com/api-keys
# puis à mettre dans la variable d'environnement OPENAI_API_KEY
utilisateur = OpenAI()

# Appel de la méthode ChatGPT
def ChatGPT(entree: str):
    sortie = utilisateur.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user", "content":entree}],
        stream=True,
    )
    return sortie

# Application de la méthode à l'entrée utilisateur
if __name__ == "__main__":
    entree = input()
    sortie = ChatGPT(entree)
    print(sortie)
