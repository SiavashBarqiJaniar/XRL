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

from math import *
import numpy as np
#print(np.random.exponential(scale=600, size=None))

class Agent(object):

    def __init__(self, initial_packets, test, NN, beta):
        self.NN = NN
        self.throughput = 0
        self.reward_list = []
        self.action_list = []
        self.no_collisions = 0
        self.collisions = []
        self.test = test
        self.no_continuous_colls = 0

    def clc_throughput(self):
        N = 1000
        i = len(self.reward_list)
        if i < N:
            r = sum(self.reward_list)
            self.throughput = r/i
        else:
            r = sum(self.reward_list[-1000:]) # from 1000th to the last item through the last item
            self.throughput = r/N