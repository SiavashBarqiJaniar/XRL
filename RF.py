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

import csv
import os
import numpy as np

from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Visualization
from sklearn import tree
import IPython, graphviz, re, math

from matplotlib import pyplot as plt

from random import shuffle

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default='trajectories/game99.csv')
parser.add_argument("--val_size", default=0.1)
parser.add_argument("--test_size", default=0.1)
parser.add_argument("--bs", default=512)
parser.add_argument("--lr", default=0.0001)
parser.add_argument("--num_epochs", default=100)
parser.add_argument("--game", default=99)
args = parser.parse_args()

###################################
# Hyperparameters
###################################
A = 9


def get_instances(startpnt, endpnt):
    verbose = True
    dataset = []
    x_ = []
    y_ = []
    for iteration in range(startpnt, endpnt+1):
        dataset_file = f'trajectories/game{iteration}.csv'
        #dataset_file = args.data_path

        with open(dataset_file, 'r') as csvfile:
            csvreader = csv.DictReader(csvfile)

            # Extracting feature names from csv
            headers = csvreader.fieldnames

            if verbose:
                print("Headers ", len(headers), headers)

            states = []
            actions = []
            q_values = []
            rewards = []
            next_states = []
            for line in csvreader:
                states.append(list(map(int, line['state'].split('$'))))
                actions.append(list(map(int, line['action'].split('$'))))
                q_values.append(float(line['qValue']))
                rewards.append(float(line['reward']))
                next_states.append(list(map(int, line['next_state'].split('$'))))

        no_instances = len(rewards)
        A = len(actions[0])
        channels = []
        for i in range(no_instances):
            channel = [[] for i in range(A)]
            for j in range(len(states[0])):
                channel[j%A].append(states[i][j])
            chs = []
            for ch in channel:
                if ch.count(-2) > 0:
                    chs.append( [1, 0, 0, 0, 1, 1, 1] ) # disrupted
                elif ch.count(-1) > 0:
                    chs.append( [0, 1, 0, 0, 1, 0, 0] ) #'occupied by the jammer')
                    #print('*** -1', ch.count(-1), ch)
                elif ch.count(1) == len(ch):
                    chs.append( [0, 0, 1, 0, 0, 1, 0] ) #'occupied by the user')
                    #print('*** 1', ch.count(1), ch)
                elif ch.count(0) == len(ch):
                    chs.append( [0, 0, 0, 1, 0, 0, 1] ) #'idle')
                    #print('*** 0', ch)
            assert len(chs) == len(channel)
            channels.append( np.concatenate(chs) )
            #print('*** result:', chs)
        
        features = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        output = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        feature_size = len(features)
        output_size = len(output)
        if verbose:
            print("Features ", len(features), features)
            print("Output ", output)

        # Creating dataset
        for i in range(no_instances):
            x = channels[i]
            y = actions[i].index(1)
            x_.append(x)
            y_.append(y)
            #dataset.append((x, y))

        if verbose:
            print("Complete dataset", len(x_), len(y_), x_[0])
        
        verbose = False
    
    return x_, y_
    
def train(x, y):
    model = RandomForestClassifier(verbose=0, max_features=1) # max_features=1

    imp = {}
    for i in range(200):
        # Shuffling instances
        z = list(zip(x, y))
        shuffle(z)
        x_, y_ = zip(*z)

        # fitting the model
        model.fit(x_, y_)

        # get importance
        importance = model.feature_importances_
        if i == 0:
            imp = importance.copy()
        else:
            imp += importance.copy()
    temp = imp.copy()/200

    return model, temp

def draw_tree(t, col_names, size=9, ratio=0.5, precision=3):
    """ Draws a representation of a random forest in IPython.
    Parameters:
    -----------
    t: The tree you wish to draw
    df: The data used to train the tree. This is used to get the names of the features.
    """
    s = tree.export_graphviz(t, out_file=None, feature_names=col_names, filled=True,
                      special_characters=True, rotate=True, precision=precision)
    IPython.display.display(graphviz.Source(re.sub('Tree {',
       f'Tree {{ size={size}; ratio={ratio}',s)))

if __name__ == '__main__':
    """
    HIPER-PARAMETERS
    """
    iterations = 5
    test = False
    scenario = "I"

    """
    ###### Accuracy
    x_test, y_test = get_instances(8, 9)
    print('data:', len(x_test), len(y_test))
    accuracy = mdl.score(x_test, y_test)
    print('Accuracy: %.2f' % (accuracy*100))
    """
    
    ### Plotting the Tree
    if False:
        model = RandomForestClassifier(criterion="entropy")
        for j in range(iterations):
            x, y = get_instances(iterations-j-1, iterations-j-1)
            for i in range(200):
                # Shuffling instances
                z = list(zip(x, y))
                shuffle(z)
                x, y = zip(*z)

                #fitting the tree model
                model.fit(x, y)
            print('data:', len(x), len(y))
    
    importances = {}
    for m in range(10):
        print('####################')
        print(f'#    iteration {m+1}')
        print('####################')
        if True:
            x, y = get_instances(50, 51) #18+ iterations-1)
            print('data:', len(x), len(y))
            model, imps = train(x, y)

        imps = model.feature_importances_
        for i,v in enumerate(imps):
            print('Feature: %0d, Score: %.5f' % (i,v))

        no_conditions = int(len(x[0])/A)
        features = []
        for i in range(A*no_conditions):
            if i%no_conditions == 0:
                features.append(f'Ch{int(i/no_conditions)+1} Collision')
            if i%no_conditions == 1:
                features.append(f'Ch{int(i/no_conditions)+1} Jammed')
            if i%no_conditions == 2:
                features.append(f'Ch{int(i/no_conditions)+1} Successful Tx')
            if i%no_conditions == 3:
                features.append(f'Ch{int(i/no_conditions)+1} Idle')
            if i%no_conditions == 4:
                features.append(f'Ch{int(i/no_conditions)+1} Collision/Jammed')
            if i%no_conditions == 5:
                features.append(f'Ch{int(i/no_conditions)+1} Collision/Successful')
            if i%no_conditions == 6:
                features.append(f'Ch{int(i/no_conditions)+1} Collision/Idle')

        fig = plt.figure(1)
        ax = fig.add_axes([.1, .25, .8, .65])
        ax.bar([m for m in range(len(imps))], imps)
        temp = 0
        ax.set_xticks(np.arange(len(features)))
        ax.set_xticklabels(features)
        for tick in ax.get_xticklabels():
            tick.set_rotation(315)
            tick.set_visible(True)
            #print(features[temp])
            #tick.set_text(features[temp])
            temp += 1
        #plt.show()
        fig.savefig('report/' + scenario + '/FI_' + scenario + '_1.png')

        if A == 3:
            fig = plt.figure(2+m)#, figsize=(12,5)) # figsize=(7, 10)
        else:
            fig = plt.figure(2+m, figsize=(12,5))
        ax = fig.add_axes([.15, .1, .8, .8])
        importances[m] = []
        for i in range(A):
            importances[m].append(sum(imps[i*no_conditions:i*no_conditions+no_conditions]))
        
        ax.bar([k for k in range(len(importances[m]))], importances[m])
        temp = 0
        ax.set_xticks(np.arange(A))
        ax.set_ylabel('Importance', fontsize=20)
        ax.set_xticklabels(['Ch. 1', 'Ch. 2', 'Ch. 3', 'Ch. 4', 'Ch. 5', 'Ch. 6', 'Ch. 7', 'Ch. 8', 'Ch. 9'])
        for tick in ax.get_xticklabels():
            tick.set_visible(True)
            tick.set_fontsize(20)
            #tick.set_rotation(300)
            #print(features[temp])
            #tick.set_text(features[temp])
            temp += 1
        for tick in ax.get_yticklabels():
            tick.set_fontsize(20)
            #tick.set_rotation(45)
        fig.savefig('report/' + scenario + '/FI_' + scenario + f'_{2+m}.jpeg')
        #plt.show()

        #draw_tree(mdl, features, precision=3)
        #tree_text = tree.export_text(model2)
        #print(tree_text)
        """
        estimator = model.estimators_[5]
        fig = plt.figure(figsize=(80,80))
        tree.plot_tree(estimator, feature_names=features, filled=True, rounded=True)#, proportion=True)
        fig.savefig('decision_tree.png')
        """
        #plt.show()
    
    #channels_imp = np.loadtxt(f'report/' + scenario + '/FI_' + scenario + '_20.txt')

    if True:
        channels_imp = [0 for m in range(A)]
        for key in importances:
            for j in range(A):
                channels_imp[j] += (importances[key][j]/10)
        with open("report/" + scenario + "/FI_" + scenario + "_20.txt", "w") as f:
            for item in channels_imp:
                f.write(str(item) + '    ')

    if A == 3:
        fig = plt.figure(20)#, figsize=(12,5)) # figsize=(7, 10)
    else:
        fig = plt.figure(20, figsize=(12,5))
    ax = fig.add_axes([.15, .1, .8, .8])

    print(np.shape(channels_imp))
    print(channels_imp)
    ax.bar([m for m in range(A)], channels_imp)
    temp = 0
    ax.set_xticks(np.arange(A))
    ax.set_ylabel('Importance', fontsize=20)
    ax.set_xticklabels(['Ch. 1', 'Ch. 2', 'Ch. 3', 'Ch. 4', 'Ch. 5', 'Ch. 6', 'Ch. 7', 'Ch. 8', 'Ch. 9'])
    for tick in ax.get_xticklabels():
        tick.set_visible(True)
        tick.set_fontsize(20)
        #tick.set_rotation(300)
        #print(features[temp])
        #tick.set_text(features[temp])
        temp += 1
    for tick in ax.get_yticklabels():
        tick.set_fontsize(20)
        #tick.set_rotation(45)
    fig.savefig('report/' + scenario + '/FI_' + scenario + '_20.jpeg')