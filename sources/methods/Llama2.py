from os import listdir
from llama_cpp import Llama

##########################################################
############ APPLICATION A TOUS LES DOCUMENTS ############
##########################################################
def folderToX(inputFolder, textToX):
    for document in listdir(inputFolder):
        f = open(inputFolder + "/" + document, "r")
        textToX(f.read(), document)
        f.close()

##########################################################
######################## LLAMA 2 #########################
##########################################################
llamaBinName = "llama-2-7b-chat.Q5_K_M.gguf"
LLM_PATH = "./Llama-2/" + llamaBinName
LLM = Llama(model_path=LLM_PATH, verbose=False)

# Default long LLaMA chat default prompt, prepended with "Answer in French".
defaultPrompt = "Answer in French. You are a helpful, respectful and honest assistant. Always answer as helpfully "      + \
                "as possible, while being safe. Your answers should not include any harmful unethical, racist, sexist, " + \
                "toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and "     + \
                "positive in nature. If a question does not make any sense, or is not factually coherent, explain why "  + \
                "instead of answering something not correct. If you don't know the answer to a question, please don't "  + \
                "share false information."

def llama2question(prompt, docs, texts, output_format):
    messages = [
            {"role": "system", "content": defaultPrompt},
            {"role": "user", "content": prompt}
    ]

    for m in range(len(docs)):
        messages.append({"role": "user", "content": docs[m] + ":\n" + " ".join(texts[m])})

    return LLM.create_chat_completion(
        messages = messages,
        response_format=output_format,
        max_tokens=None
    )['choices'][0]['message']['content']

###########################################################
####################### APPLICATION #######################
###########################################################
dataCouples = [
        ("input-data/Raw+Babelfy/prelinked", "output-data/Raw+Llama2"),
        ("input-data/Raw+Babelfy/equivalent", "output-data/Raw+Babelfy+Llama2"),
        ("input-data/Raw+RNNTagger/punctuationClean", "output-data/Raw+RNNTagger+Llama2"),
        ("input-data/Raw+RNNTagger+Babelfy/equivalent/punctuationClean", "output-data/Raw+RNNTagger+Babelfy+Llama2/punctuationClean"),
        ("input-data/Raw+RNNTagger+Babelfy/equivalent/keepPunctuation", "output-data/Raw+RNNTagger+Babelfy+Llama2/keepPunctuation"),
        ("input-data/Raw+TreeTagger/lemmatized/keepStopData", "output-data/Raw+TreeTagger+Llama2/keepStopData"),
        ("input-data/Raw+TreeTagger/lemmatized/throwStopClasses", "output-data/Raw+TreeTagger+Llama2/throwStopClasses"),
        ("input-data/Raw+TreeTagger/lemmatized/throwStopWords", "output-data/Raw+TreeTagger+Llama2/throwStopWords"),
        ("input-data/Raw+TreeTagger+Babelfy/equivalent/keepStopData", "output-data/Raw+TreeTagger+Babelfy+Llama2/keepStopData"),
        ("input-data/Raw+TreeTagger+Babelfy/equivalent/throwStopClasses", "output-data/Raw+TreeTagger+Babelfy+Llama2/throwStopClasses"),
        ("input-data/Raw+TreeTagger+Babelfy/equivalent/throwStopWords", "output-data/Raw+TreeTagger+Babelfy+Llama2/throwStopWords"),
    ]

idDictionaryPaths = [
    "",
    "input-data/Raw+Babelfy/dictionary/bn_ids.csv",
    "",
    "input-data/Raw+RNNTagger+Babelfy/dictionary/punctuationClean/bn_ids.csv",
    "input-data/Raw+RNNTagger+Babelfy/dictionary/keepPunctuation/bn_ids.csv",
    "",
    "",
    "",
    "input-data/Raw+TreeTagger+Babelfy/dictionary/keepStopData/bn_ids.csv",
    "input-data/Raw+TreeTagger+Babelfy/dictionary/throwStopClasses/bn_ids.csv",
    "input-data/Raw+TreeTagger+Babelfy/dictionary/throwStopWords/bn_ids.csv",
]

descriptions = {
    "Les textes suivants ont été filtrés via nltk en python. ",
    "Les textes suivants ont été filtrés via nltk en python, puis traités par Babelfy. ",
    "Les textes suivants ont été filtrés via nltk en python, puis via RNNTagger. ",
    "Les textes suivants ont été filtrés via nltk en python, puis via RNNTagger, puis traités par Babelfy. ",
    "Les textes suivants ont été filtrés via nltk en python, puis via RNNTagger, puis traités par Babelfy. ",
    "Les textes suivants ont été filtrés via nltk en python, puis via TreeTagger. ",
    "Les textes suivants ont été filtrés via nltk en python, puis via TreeTagger. ",
    "Les textes suivants ont été filtrés via nltk en python, puis via TreeTagger. ",
    "Les textes suivants ont été filtrés via nltk en python, puis via TreeTagger, puis traités par Babelfy. ",
    "Les textes suivants ont été filtrés via nltk en python, puis via TreeTagger, puis traités par Babelfy. ",
    "Les textes suivants ont été filtrés via nltk en python, puis via TreeTagger, puis traités par Babelfy. ",
}

score_json = {
    "type": "json_object",
    "schema": {
        "topics": [[{"word": "string", "score": "float" }]],
        "documents": [{"document": "string", "topics": ["float"]}]
    }
}
simple_json = {
    "type": "json_object",
    "schema": {
        "topics": [[{"word": "string"}]],
        "documents": [{"document": "string", "topics": ["float"]}]
    }
}

promptBasic = "Résume les textes suivants:"

if __name__ == '__main__':
    print("Llama2...", flush=True)
    l_data = len(dataCouples)
    for i in range(l_data):
        inputFolder, outputFolder = dataCouples[i]
        print(f"Llama2 - {inputFolder}... ({i+1}/{l_data})", flush=True)
        texts, docs = [], []
        def textToCorpus(text, document):
            docs.append(document)
            texts.append(text.splitlines())

        folderToX(inputFolder, textToCorpus)

        # Parameters
        formats = [None] #, score_json, simple_json]
        useDescription = [False, True]
        prompts = [promptBasic]

        for j in range(len(formats)):
            outputFormat = formats[j]
            for k in range(len(useDescription)):
                prompt_intro = descriptions[i] if useDescription[k] else ""
                for l in range(len(prompts)):
                    prompt_core = prompts[l]

                    fileName = str(j) + "_" + str(k) + "_" + str(l)
                    extention = ".txt" if j == 0 else ".json"

                    print(fileName + extention, flush=True)

                    f = open(outputFolder + "/" + fileName + ".prompt", "w")
                    f.write(prompt_intro + prompt_core + "[...]" + "\n")
                    f.close()

                    prompt = prompt_intro + prompt_core

                    f = open(outputFolder + "/" + fileName + extention, "w")
                    f.write(llama2question(prompt, docs, texts, outputFormat))
                    f.close()