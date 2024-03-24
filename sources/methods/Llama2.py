# Chemin du binaire Llama (téléchargeable ici: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)
LLM_PATH = "./Llama-2/llama-2-7b-chat.Q5_K_M.gguf"

import sys, os
from llama_cpp import Llama

LLM = Llama(model_path=LLM_PATH)

prompt = "Q: What are the names of the days of the week? A:"

# generate a response (takes several seconds)
output = LLM(prompt)

# display the response
print(output)
