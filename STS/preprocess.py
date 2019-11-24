from lib.Preprocessing.reader import CorpusReader
from lib.Preprocessing.prepprocessing import Preprocessing

if __name__ == "__main__":
    reader = CorpusReader("data")
    dev_set = Preprocessing(reader.get())
    train_set = Preprocessing(reader.get(dev=1))
    test_set = Preprocessing(reader.get(dev=2))
    dev_set.transform().store("dev")
    train_set.transform().store("train")
    test_set.transform().store("test")
