from lib.Preprocessing.reader import CorpusReader
from lib.Preprocessing.featureGenerator import Preprocessing
import os

if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    directory = os.path.join(dirname, 'data/')
    Storedirectory = os.path.join(dirname, 'PreProcessesData/')
    reader = CorpusReader(directory)
    dev_set = Preprocessing(reader.get())
    train_set = Preprocessing(reader.get(dev=1))
    test_set = Preprocessing(reader.get(dev=2))
    dev_set.transform().store(Storedirectory+"dev")
    train_set.transform().store(Storedirectory+"train")
    test_set.transform().store(Storedirectory+"test")
