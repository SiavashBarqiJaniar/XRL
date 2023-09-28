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

from environment import ENVIRONMENT
from agent import Agent
from DQN_brain import DQN
from sklearn.cluster import KMeans

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from math import *
import numpy as np
from numpy.random import shuffle
import matplotlib.pyplot as plt
import time
import csv

#==========================
#   Hyperparameters
#==========================
#frequency band = 20 MHz
#delta_f = 100 kHz # every 1ms scan 100kHz of whole spectrum which takes 200ms
#J = 30 dBm
#U = 0 dBm
Lambda = .2
T = 10#1#8#16#200
model_name = 'DNN_S'
if model_name == 'DNN_D3':
    A = 3
else:
    A = 9
N = A#200
# Remark: s and action are so similar here

noClusters = 3 # = number of classes
clusterModel = KMeans(n_clusters = noClusters)

fig = plt.figure(1)
ax = fig.add_axes([.1, .15, .8, .7])

def identifyPattern(samples):
    indexes = list(range(len(samples)))
    for i in range(10):
        temp = list(zip(indexes, samples))
        shuffle(temp)
        indexes, samples = zip(*temp)
        clusterModel.fit(samples)
    labels = clusterModel.predict(samples)
    #print(labels)
    sortedIndexes, sortedLabels = zip(*sorted(zip(indexes, labels)))
    #print(sortedIndexes)
    #print(sortedLabels)
    return sortedLabels[-1]

def dBm_watt(power_dbm):
    return 10**((power_dbm - 30)/10)

def main(time_frame, game_number, reward_max):
    #print('------------------------------------------')
    print(f'----- Start processing game {game_number} ... -----')
    #print('------------------------------------------')

    state = np.ndarray(shape=(10, N), dtype=int) # T x N(s)
    action = np.zeros(A, dtype=int) # size = A
    reward_list = []
    start = time.time()
    action_list = []
    raw_action_list = []
    q_value_list = []
    raw_state_list = []
    state_list = []
    next_state_list = []
    record_transition = []
    clusterSamples = []
    samplePatterns = []
    s = env.reset() # size = N
    
    user_period = 10
    sweep_jammer_counter = 0
    current_channel = 0
    previous_action = np.zeros(A, dtype=int)
    last_mu = 1
    channels_usage_penalty = [0 for i in range(A)]
    check = True
    check2 = [False for i in range(A)]
    rewards_history = []
    count_collisions = 0
    pattern_switch_time = 0
    flag = 1
    patternChange = 0

    for i in range(time_frame):
        if i == patternChange:
            patternChange += np.random.randint(100, 301)
            toss = np.random.randint(3)
            if toss == 0:
                model_name = 'DNN_D9'
            elif toss == 1:
                model_name = 'DNN_I'
            elif toss == 2:
                model_name = 'DNN_S'
        if i%1000==0:
            print(f'In iteration {i}', f'epsilon: {dqn_agent.epsilon}')

        #===============
        #    State
        #===============
        end_of_signal = False
        if i%10 == 9:
            end_of_signal = True
        #if i < T:
        #    state.append(s)
        #else:
        #    temp = state.copy()
        #    temp = state[1:]
        #    temp.append(s)
        #    state = temp.copy()

        #===============
        #    Action
        #===============
        if i%10 == 0:
            if i > 9:
                previous_action = action.copy()
            if i >= T*10:
                #print('i:', i)
                #print(state)
                action, q_value, q_values = dqn_agent.choose_action(state_list[-T:], i, check) # argmax(Q(shape=(1,2))) = 0 || 1
                check = False
                q_value_list.append(q_value)
                if i%1000 == 0:
                    print(f'    action {action}\n    q_values{q_values.tolist()}\n')
            elif i<T*10:
                action = np.zeros(A, dtype=int)
                action[np.random.randint(0, A)] = 1
                q_values = -10
                q_value_list.append(-10)
            action_list.append(action)
        raw_action_list.append(action)

        #=============================
        #   Jammer Action
        #=============================
        if model_name == "DNN_D":
            #jammer_action[int((i%9000)/1000)] = -1
            for j in range(A):
                if (i%1000) < 800:
                    if j%2 == 0:
                        jammer_action[j] = -1
                else:
                    if j%2 == 1:
                        jammer_action[j] = -1
            
            jammer_action = np.ones(A, dtype=int)
            jammer_action *= -1
            jammer_action[6] = 0
        
        
        ########################
        #   Intelligent
        ########################
        if model_name == "DNN_temp":
            if i%900 < 100:
                jammer_action = np.zeros(A, dtype=int)
        










        ########################
        #   Intelligent
        ########################
        if model_name == "DNN_I":
            if i%100 == 0:
                jammer_action = np.zeros(A, dtype=int)
                maxx = [0 for x in range(A)]
                for a in action_list[-10:]:
                    maxx[np.where(a==1)[0][0]] += 1
                for x in range(6):
                    if max(maxx) != 0:
                        jammer_action[np.argmax(maxx)] = -1
                        maxx[np.argmax(maxx)] = 0
                    else:
                        break
        
        ########################
        #   Dynamic
        ########################
        if model_name == "DNN_D9":
            jammer_action = np.ones(A, dtype=int)*(-1)
            if (i%100) < 50:
                jammer_action[6] = jammer_action[7] = jammer_action[8] = 0 #jammer_action[0] = 0
            else:
                jammer_action[2] = jammer_action[3] = jammer_action[4] = 0 #jammer_action[2] = 0

        ########################
        #   Dynamic
        ########################
        if model_name == "DNN_D3":
            jammer_action = np.ones(A, dtype=int)*(-1)
            if (i%100) < 50:
                jammer_action[0] = 0
            else:
                jammer_action[2] = 0

        #########################
        #   Sweep
        #########################
        if model_name == "DNN_S":
            jammer_action = np.zeros(A, dtype=int)
            if i%24 <= 6:
                jammer_action[0] = -1
            if i%24 <= 8 and i%24 >= 2:
                jammer_action[1] = -1
            if i%24 <= 10 and i%24 >= 4:
                jammer_action[2] = -1
            if i%24 <= 12 and i%24 >= 6:
                jammer_action[3] = -1
            if i%24 <= 14 and i%24 >= 8:
                jammer_action[4] = -1
            if i%24 <= 16 and i%24 >= 10:
                jammer_action[5] = -1
            if i%24 <= 18 and i%24 >= 12:
                jammer_action[6] = -1
            if i%24 <= 20 and i%24 >= 14:
                jammer_action[7] = -1
            if i%24 <= 22 and i%24 >= 16:
                jammer_action[8] = -1
        

        ###########################
        #   Comb-Sweep
        ###########################
        if model_name == "DNN_CS":
            if pattern_switch_time == 0:
                pattern_switch_time = np.random.randint(10, 21)
                flag *= -1
            if flag == 1:
                ### Comb
                jammer_action = np.ones(A, dtype=int)*(-1)
                jammer_action[2] = jammer_action[6] = 0
            else:
                ### Sweep
                jammer_action = np.zeros(A, dtype=int)
                if i%24 <= 6:
                    jammer_action[0] = -1
                if i%24 <= 8 and i%24 >= 2:
                    jammer_action[1] = -1
                if i%24 <= 10 and i%24 >= 4:
                    jammer_action[2] = -1
                if i%24 <= 12 and i%24 >= 6:
                    jammer_action[3] = -1
                if i%24 <= 14 and i%24 >= 8:
                    jammer_action[4] = -1
                if i%24 <= 16 and i%24 >= 10:
                    jammer_action[5] = -1
                if i%24 <= 18 and i%24 >= 12:
                    jammer_action[6] = -1
                if i%24 <= 20 and i%24 >= 14:
                    jammer_action[7] = -1
                if i%24 <= 22 and i%24 >= 16:
                    jammer_action[8] = -1
            if end_of_signal:
                pattern_switch_time -= 1

        ###########################
        #   SSSSSSSSSSSSSweep
        ###########################
        if model_name == "DNN_SS":
            if sweep_jammer_counter%4 == 0:
                if current_channel == 9:
                    jammer_action = np.zeros(9, dtype=int)
                    jammer_action[0] = -1
                    current_channel = -1
                    sweep_jammer_counter += 1
                else:
                    jammer_action = np.zeros(9, dtype=int)
                    jammer_action[current_channel] = -1
                current_channel += 1
                sweep_jammer_counter += 1
            else:
                jammer_action = np.zeros(9, dtype=int)
                if current_channel == 0:
                    jammer_action[current_channel] = -1
                elif current_channel == 9:
                    jammer_action[current_channel-1] = -1
                else:
                    jammer_action[current_channel] = -1
                    jammer_action[current_channel-1] = -1
                sweep_jammer_counter += 1

        #===============
        #    Step
        #===============
        mu, next_state = env.step(action, Lambda, jammer_action)
        state[i%10] = next_state.copy()
        raw_state_list.append(next_state)
        if end_of_signal:
            state_list.append(state.copy())
            if i >= 999:
                clusterSamples.append( np.concatenate(raw_state_list[-1000:]) )
                if len(clusterSamples) >= noClusters:
                    pattern = identifyPattern(clusterSamples[-20:])
                    for k in range(10):
                        samplePatterns.append(pattern)
        last_mu *= mu

        #===============
        #    Reward Shaping
        #===============
        if end_of_signal:
            switched = 0
            action_idx = np.where(action==1)[0][0]
            if (action == previous_action).all():
                switched = 0
            #    channels_usage_penalty[action_idx] = min(channels_usage_penalty[action_idx] + .3, 1)
            reward = last_mu*reward_max - Lambda*switched #*.2
            if True:
                if True: #reward == 1 or check2[0]:
                    if True: #check2[0] and reward != 1:
                        with open(f'hist/reward_1_{game_number}.txt', 'a') as f:
                            f.write('t = ' + str(i) + ' r ' + str(reward) + ' action ' + str(action) + ' Q ' + str(q_values-q_values%.001) + '\n')
                        check2[0] = False
                    else:
                        with open(f'hist/reward_1_{game_number}.txt', 'a') as f:
                            f.write('time ' + str(i) + ' reward ' + str(reward) + ' action ' + str(action) + ' Q ' + str(q_value) + '\n')
                        check2[0] = True
            """
            reward = last_mu - channels_usage_penalty[action_idx]
            channels_usage_penalty[action_idx] = min(channels_usage_penalty[action_idx] + .4, 1)
            for j in range(A):
                channels_usage_penalty[j] = max(channels_usage_penalty[j] - .1, 0)
            """
            
            if last_mu == 0:
                count_collisions += 1
            
            rewards_history.append(reward)
            for j in range(10):
                reward_list.append(reward)
            last_mu = 1


        next_state_list.append(next_state)
        if end_of_signal and i>=T*10 + 10 - 1:
            dqn_agent.store_transition(state_list[-(T+1):-1], action, reward, state_list[-T:]) # SO IMPORTANT!!!! stores the trajectory
            
            #=======================================
            #   converting trajectories to strings
            #=======================================
            state_str = ''
            #if (state_list[-2] == state_list[-1]).all():
            #    state_str += '### '
            for feature in np.concatenate(state_list[-2]):
                state_str += str(feature) + '$'
            state_str = state_str[:-1]

            action_str = ''
            for feature in action:
                action_str += str(feature) + '$'
            action_str = action_str[:-1]
            
            next_state_str = ''
            for feature in np.concatenate(state_list[-1]): #state):
                next_state_str += str(feature) + '$'
            next_state_str = next_state_str[:-1]

            record_transition.append({'state': state_str, 'action': action_str, 'qValue': q_value, 'reward': reward,
                            'next_state': next_state_str})

        if not test:
            if i >= T*10+10-1 and end_of_signal: # and i%100==0:
                dqn_agent.learn()

        s = next_state
    
    #######################
    #   the end of the game
    #######################
    with open(f'./trajectories/game{game_number}.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=record_transition[0].keys(), lineterminator='\n')
        writer.writeheader()
        for row_dict in record_transition:
            writer.writerow(row_dict)

    print('game_number', game_number)
    with open(f'rewards/reward{game_number}.txt', 'w') as f:
        for i in reward_list:
            f.write(str(i) + '   ')
    with open(f'rewards/action{game_number}.txt', 'w') as f:
        for i in raw_action_list:
            f.write(str(np.where(i==1)[0][0]) + '   ')
    with open(f'rewards/jammerAction{game_number}.txt', 'w') as f:
        for i in raw_state_list:
            temp = ''
            temp2 = '-100 '
            temp3 = 9
            ## for the very first time step:
            for j in np.where(i==-1)[0]:
                temp += str(j) + ' '
            for j in np.where(i==-2)[0]:
                temp += str(j) + ' '
            if len(np.where(i==-1)[0]) + len(np.where(i==-2)[0]) < temp3:
                for x in range(temp3- ( len(np.where(i==-1)[0]) + len(np.where(i==-2)[0]) )):
                    temp += temp2
            f.write(temp[:-1] + '\n')
    with open(f'rewards/state{game_number}.txt', 'w') as f:
        for i in raw_state_list:
            for j in i:
                f.write(str(j) + '   ')
            f.write('\n')
    with open(f'rewards/clusters{game_number}.txt', 'w') as f:
        for i in samplePatterns:
            f.write(str(i) + '   ')








    
    print('Time elapsed:', time.time()-start, 'seconds')
    #print('------------------------------------------')
    print('---------- ... End of processing ----------')
    print('------------------------------------------')
    if not test:
        dqn_agent.model.save(f'{model_name}.h5')
        #tf.keras.experimental.export_saved_model(dqn_agent.model, 'DNN.h5')
    
    throughput = 1 - count_collisions/(time_frame/10)
    print('Throughput: ', throughput)
    if False:#test:
        with open(f'report/{model_name[-2:]}.txt', 'a') as f:
            f.write(f'game # {game_number}:' + str(throughput) + '\r\n')
    return throughput

from plot import draw

if __name__ == "__main__":
    #==========================
    #   Hyperparameters
    #==========================
    test = True
    time_frame = 10000
    iterations = 10
    #==========================

    #env = ENVIRONMENT(state_shape=10*N*T, action_size=A, next_state_shape=10*N)
    
    last_games_throughput_list = []
    for t in range(1): # >1 just when finding optimum value for a variable
        T = 10
        env = ENVIRONMENT(state_shape=(10*T, N), action_size=A, next_state_shape=10*N)
        print('####################')
        print('#     ', t+1)
        print('####################')
        dqn_agent = DQN(env.state_shape,
                        env.n_actions,
                        env.next_state_shape,
                        memory_size=1,#5,#128#1000
                        replace_target_iter=10,#100#20,#10#
                        batch_size=1,#5,#32
                        learning_rate=.01,#.01
                        gamma= .1,#.1#.9
                        epsilon=1,#1.1
                        epsilon_min=0.05,#0.05,#.05
                        epsilon_max=.9,
                        epsilon_decay=0.9985,#93,
                        test = test,
                        model_name = model_name,
                        )

        if test:
            temp = 1
        else:
            temp = iterations

        for i in range(temp):
            if test:
                thr = main(time_frame=time_frame, game_number=50+i, reward_max = t)
            else:
                thr = main(time_frame=time_frame, game_number=i+temp*t, reward_max = 1)
        last_games_throughput_list.append(thr)
    
    print(last_games_throughput_list)
    #fig = plt.figure(1)
    #ax = fig.add_axes([.1, .1, .8, .8])
    #ax.plot(last_games_throughput_list, color='r', lw=3)
    #fig.savefig('report/last_games_throughput_list_batch_size.jpeg')