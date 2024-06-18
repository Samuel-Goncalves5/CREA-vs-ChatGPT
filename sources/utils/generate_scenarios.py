import os

def getScenarios():
    f = open("./scenarios.csv", "r")
    content = f.read().splitlines()
    f.close()

    scenarios = [scenario.split(";") for scenario in content]
    return [(scenario[0], scenario[1:]) for scenario in scenarios]

if __name__ == '__main__':
    subfolders = ["Raw+Babelfy+CREA", "Raw+Babelfy+LDA", "Raw+Babelfy+Llama2",
                "Raw+LDA", "Raw+Llama2",
                "Raw+RNNTagger+Babelfy+CREA", "Raw+RNNTagger+Babelfy+LDA", "Raw+RNNTagger+Babelfy+Llama2",
                "Raw+RNNTagger+CREA", "Raw+RNNTagger+LDA", "Raw+RNNTagger+Llama2",
                "Raw+TreeTagger+Babelfy+CREA", "Raw+TreeTagger+Babelfy+LDA", "Raw+TreeTagger+Babelfy+Llama2",
                "Raw+TreeTagger+LDA", "Raw+TreeTagger+Llama2"]

    for s,_ in getScenarios():
        try:
            os.mkdir("./output/" + s)
        except:
            pass

        try:
            os.mkdir("./output/" + s + "/data")
        except:
            pass

        for sub in subfolders:
            try:
                os.mkdir("./output/" + s + "/data/" + sub)
            except:
                pass
        
        try:
            os.mkdir("./output/" + s + "/evaluations")
        except:
            pass