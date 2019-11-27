from lib.ModelTools.features import Features
import pandas as pd
import os

if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    directory = os.path.join(dirname, 'PreProcessesData/')
    Storedirectory = os.path.join(dirname, 'Features/')
    dev_set = pd.read_pickle("{}dev".format(directory))
    train_set = pd.read_pickle("{}train".format(directory))
    test_set = pd.read_pickle("{}test".format(directory))
    Features(dev_set).generate().store(Storedirectory+"dev")
    Features(train_set).generate().store(Storedirectory+"train")
    Features(test_set).generate().store(Storedirectory+"test")
