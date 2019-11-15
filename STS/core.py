from reader  import CorpusReader
from pre_processing import Preprocessing
if __name__ == "__main__":
    reader = CorpusReader("data")
    dev_set = reader.get()
    train_set = reader.get(dev=1)
    test_set = reader.get(dev=2)
    print(Preprocessing(dev_set).getAllSentences())
    # print(dev_set.head(),train_set.head())

