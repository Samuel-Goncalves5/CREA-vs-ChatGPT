# Chemin du binaire Llama (téléchargeable ici: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)
LLM_PATH = "./Llama-2/llama-2-7b-chat.Q5_K_M.gguf"

import sys, os
from llama_cpp import Llama

LLM = Llama(model_path=LLM_PATH, verbose=False)

# Default long LLaMA chat default prompt, prepended with "Answer in French".
defaultPrompt = "Answer in French. You are a helpful, respectful and honest assistant. Always answer as helpfully "      + \
                "as possible, while being safe. Your answers should not include any harmful unethical, racist, sexist, " + \
                "toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and "     + \
                "positive in nature. If a question does not make any sense, or is not factually coherent, explain why "  + \
                "instead of answering something not correct. If you don't know the answer to a question, please don't "  + \
                "share false information."

prompt = "Quels sont les sept jours de la semaine?"

output = LLM.create_chat_completion(
      messages = [
          {"role": "system", "content": defaultPrompt},
          {"role": "user", "content": prompt}
      ],
      max_tokens=None
)

# display the response
print(prompt + "\n")
print(output['choices'][0]['message']['content'])