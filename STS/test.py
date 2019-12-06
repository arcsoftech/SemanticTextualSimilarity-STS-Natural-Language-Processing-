from lib.Preprocessing.featureGenerator import Preprocessing
import pandas as pd
import os

if __name__ == "__main__":
    print("Please enter sample sentence")
    sentence = input()
    dirname = os.path.dirname(__file__)
    directory = os.path.join(dirname, 'test/')
    data = [[sentence, ""]]  
    df = pd.DataFrame(data, columns = ['Sentence1', 'Sentence2']) 
    features =Preprocessing(df).transform()
    features.store(directory+"test.csv",pickle=False)
    print(features.data)
 
