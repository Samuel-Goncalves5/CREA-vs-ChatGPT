import os

ignore = ["./preprocessing/TreeTagger"]

def printall(folder, limit=False):
    if (folder in ignore) or (limit and len(os.listdir(folder))) >= 20:
        print(folder + "/...")
    
    else:
        for f in os.listdir(folder):
            f = folder + "/" + f
            if (os.path.isdir(f)):
                printall(f, True)
            else:
                print(f)

printall(".")