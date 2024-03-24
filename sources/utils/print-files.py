import os

def printall(folder, limit=False):
    if limit and len(os.listdir(folder)) > 10:
        print(folder + "/...")
    
    else:
        for f in os.listdir(folder):
            f = folder + "/" + f
            if (os.path.isdir(f)):
                printall(f, True)
            else:
                print(f)

printall(".")
