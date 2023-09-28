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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gn", default=0)
args = parser.parse_args()

game_number = args.gn

### calculate throughput
def cal_throughput(max_iter, N, reward):
	temp_sum = 0
	throughput = np.zeros(max_iter)
	for i in range(max_iter):
		if i < N:
			temp_sum += reward[i]
			throughput[i] = temp_sum / (i+1)
		else:
			temp_sum  += reward[i] - reward[i-N]
			throughput[i] = temp_sum / N
	return throughput

def draw(gn=0):
	#game_number = max(gn, args.gn)
	my_agent_throughputs = {}
	rewards = []
	actions = []
	states = []
	clusters = []
	agent_throughputs = []
	sum_throughputs = {}
	N = 9
	A = 9

	rewards.append(np.loadtxt(f'rewards/reward{game_number}.txt'))
	actions.append(np.loadtxt(f'rewards/action{game_number}.txt'))
	states.append(np.loadtxt(f'rewards/jammerAction{game_number}.txt'))
	clusters.append(np.loadtxt(f'rewards/clusters{game_number}.txt'))
	max_iter = len(states[0])
	agent_throughputs.append(cal_throughput(len(rewards[0]), N, rewards[0]))

	x = np.linspace(0, max_iter, max_iter)

	
	sum_throughputs  = sum(agent_throughputs)

	collisions = []
	temp = []
	for i in range(len(actions[0])):
		#print("*** 51:", actions[0][i], states[0][i], actions[0][i] == states[0][i])
		if (actions[0][i] == states[0][i]).any():
			collisions.append(actions[0][i].copy())
			#actions[0][i] = states[0][i] = -9
		else:
			collisions.append(-9)

	for i in range(A):
		print(f'Channel {i+1}:', np.count_nonzero(actions[0] == i))

	plt.figure(2)
	plt.subplot(2, 1, 1)
	plt.plot(actions[0]+np.ones_like(actions[0]), '.', color='green', lw=1, label='User')
	plt.plot(states[0]+np.ones_like(states[0]), '.', color='blue', lw=1, label='Jammer')
	plt.plot(collisions+np.ones_like(collisions), '.', color='red', lw=1, label='Collision')

	plt.subplot(2, 1, 2)
	plt.plot(clusters[0], '.', color='red', lw=1, label='clusters')

	plt.show()


	fig = plt.figure(1)
	ax = fig.add_axes([.1, .55, .8, .7])
	ax2 = fig.add_axes([.1, .15, .8, .45])
	#ax.axis('off')
	ax.set_ylim(0, A+1)
	ax.set_ylabel('Channel', fontsize=26)
	plt.xlabel('Time', fontsize=26)
	ax.grid()
	ax.set_yticks(np.arange(1, A+1, 1))
	#ax.set_yticklabels([1, 2, 3], ['Channel 1', 'Channel 2', 'Channel 3'])
	count = 0
	for tick in ax.get_xticklabels():
		tick.set_fontsize(26)
		tick.set_visible(True)
	for tick in ax.get_yticklabels():
		tick.set_fontsize(26)
		count += 1
		tick.set_text(f'Channel{count}')
	ax.grid(False)
	ax.plot(actions[0]+np.ones_like(actions[0]), '.', color='green', lw=1, label='User')
	ax.plot(states[0]+np.ones_like(states[0]), '.', color='blue', lw=1, label='Jammer')
	ax.plot(collisions+np.ones_like(collisions), '.', color='red', lw=1, label='Collision')
	ax2.plot(clusters[0], '.', color='red', lw=1, label='clusters')
	handles, labels = ax.get_legend_handles_labels()
	handles = [handles[0], handles[1], handles[-1]]
	labels = ['User', 'Jammer', 'Collision']
	ax.legend(handles, labels, loc='lower left', ncol=4, bbox_to_anchor=(0,1), fontsize=26)
	plt.show()

if __name__ == "__main__":
	draw()



















