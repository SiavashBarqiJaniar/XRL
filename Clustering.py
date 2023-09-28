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

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from random import shuffle
import pandas as pd

A = 9
resolution = 0
noClusters = 5 # = number of classes
model = KMeans(n_clusters = noClusters)

fig = plt.figure(1)
ax = fig.add_axes([.1, .15, .8, .7])

def identifyPattern(samples):
    indexes = list(range(len(samples)))
    for i in range(10):
        temp = list(zip(indexes, samples))
        shuffle(temp)
        indexes, samples = zip(*temp)
        model.fit(samples)
    labels = model.predict(samples)
    #print(labels)
    sortedIndexes, sortedLabels = zip(*sorted(zip(indexes, labels)))
    #print(sortedIndexes)
    #print(sortedLabels)
    return sortedLabels[-1]


for k in range(1):
    samples = []
    samplePatterns = []
    resolution = 3000 # 3000 was optimum = 3 seconds
    for j in range(0, 8):

## Load the Sate
        states = []
        states.append(np.loadtxt(f'rewards/state{j}.txt'))
        states = states[0]
        print('state shape:', np.shape(states))

## Generate Samples
        samplesGenerationStep = 100 # every 1s create a sample
        noSamples = int(   (  (len(states) - 3000) / samplesGenerationStep  )   +   1   )
        for i in range(noSamples):
            position = samplesGenerationStep*i
            samples.append(    np.concatenate(  states[position:resolution + position]  )    ) # past 1000ms of the observation
            print('*** i:', j*noSamples + i)
            if len(samples) >= noClusters:
                pattern = identifyPattern(samples[-20:])
                for k in range(samplesGenerationStep*10):
                    samplePatterns.append(pattern)
        print(len(samples), np.shape(samples[-1]))

## noClusters == 2?
    if noClusters == 2:
        # creating dump data
        for j in range(4*noSamples):
            dummyData = []
            for j in range(resolution):
                dummySample = [random.randint(-1, 1) for m in range(A)]
                # with 50% prob one of the elements is 1
                if random.randint(0, 2) == 1:
                    dummySample[random.randint(0, A)] = 1
                dummyData.append(dummySample)
            samples.append(    np.concatenate(    dummyData    )    )
    
## Storing Samples
    dfSamples = pd.DataFrame(data=samples)
    dfSamples.to_csv('unsupervisedSamples.csv')

    print('Identified Patterns:', len(samplePatterns))
    #dfSortedLabels = pd.DataFrame(data=sortedLabels)
    #dfSortedLabels.to_csv('classes.csv')
    dfSamplePatterns = pd.DataFrame(data=samplePatterns)
    dfSamplePatterns.to_csv('recognizedPatterns.csv')


    samples_ = np.array(samples)
    xs = samples_[:, 0]
    ys = samples_[:, 4]
    zs = samples_[:, 2]

    #plt.scatter(xs, ys, zs, c = labels)

    #ax.plot(sortedLabels)

"""
ax.set_xticks(np.arange(1, 24, 1))

for tick in ax.get_xticklabels():
    #tick.set_fontsize(26)
    tick.set_visible(True)
"""
plt.show()