"""
Copyright 2023 Siavash Barqi Janiar

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from pandas import read_csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from matplotlib import pyplot
import numpy as np

A = 9

# load the dataset
def load_dataset(filename1, filename2):
    # load the dataset as a pandas DataFrame
    data = read_csv(filename1, header=None)
    yData = read_csv(filename2, header=None)
    # retrieve numpy array
    dataset = data.values
    yDataset = yData.values
    # split into input (X) and output (y) variables
    X = dataset[1:, 1:]
    y = yDataset[1:, 1:]
    print('data size:', np.shape(X), np.shape(y))
    # format all fields as string
    #X = X.astype(str)
    return X, y

# feature selection
def select_features(X_train, y_train, X_test):
	fs = SelectKBest(score_func=chi2, k='all')
	fs.fit(X_train + np.ones_like(X_train), y_train)
	X_train_fs = fs.transform(X_train)
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

# load the dataset
X, y = load_dataset('unsupervisedSamples.csv', 'classes.csv')
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# prepare input data
#X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
# prepare output data
#y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# what are scores for the features
channelScores = [0.0 for m in range(A)]
for m in range(len(fs.scores_)):
    for n in range(A):
        if m%A == n:
            if np.isnan(fs.scores_[m]):
                continue
            else:
                channelScores[n] += fs.scores_[m]
totalScores = sum(channelScores)
for i in range(A):
    channelScores[i] /= totalScores
    print('Feature %d: %f' % (  i, channelScores[i]  ))
dfScores = pd.DataFrame(fs.scores_)
dfScores.to_csv('Scores.csv')
# plot the scores
pyplot.bar([i for i in range(A)], channelScores)
pyplot.show()