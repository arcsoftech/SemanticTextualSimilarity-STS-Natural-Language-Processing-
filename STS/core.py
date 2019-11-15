from reader  import CorpusReader
if __name__ == "__main__":
    reader = CorpusReader("data")
    dev_set = reader.get()
    train_set = reader.get(dev=1)
    test_set = reader.get(dev=2)
    print(dev_set.head(),train_set.head())

    
  