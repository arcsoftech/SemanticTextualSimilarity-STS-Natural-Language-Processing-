This project aims to find the confidence level from 1 to 5 for semanitic similarity between two text document.

Group Name: OUTLIER
Group Memebers:
1. Arihant Chhajed
2. Diksha Chhabra

Download all the package from the requirement.txt
Go to STS Directory
## Step #1 Preprocessing

Execute preprocess.py to generate preprocessed data

eg python preprocess.py

## Step #2 Feature Generation
Execute generate_feature.py to generate features from data
eg python generate_feature.py

## Step #3 Model execution
Exceute main.py for model exection
argument can be as follows:-
0: Train
1: Train&Test(dev set)
2: Test (test set)
eg python main.py <argument>

Predictions can be seen in "Predictions Folder" for all the model and average of those model.

# For Feature Testing
execute test.py
 eg python test.py

