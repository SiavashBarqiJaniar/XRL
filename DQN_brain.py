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
import os
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from tensorflow.python.keras.backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))
import tensorflow_hub as hub

from tensorflow import keras
from tensorflow.keras import layers

from keras.models import Sequential, Model, load_model
#from tensorflow.keras.layers import Dense, Dropout, Input, Add, Activation, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from keras.initializers import glorot_normal

import math

class DQN:
	def __init__(self,
				state_size,
				n_actions,
				next_state_shape,
				memory_size=500,
				replace_target_iter=200,
				batch_size=32,
				learning_rate=0.01,
				gamma=0.9,
				epsilon=0.6,
				epsilon_min=0.01,
				epsilon_max=1,
				epsilon_decay=.95,
				test = False,
				model_name = 'hello world',
				):
		# hyper-parameters
		self.test = test
		self.state_size = state_size
		self.n_actions = n_actions
		self.next_state_shape = next_state_shape
		self.memory_size = memory_size
		self.replace_target_iter = replace_target_iter
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.gamma = gamma
		if test:
			self.epsilon = 0#.08 #min(0.01, epsilon)
		else:
			self.epsilon = epsilon
		self.epsilon_min = epsilon_min
		self.epsilon_max = epsilon_max
		self.epsilon_decay = epsilon_decay
		self.memory = [[] for x in range(self.memory_size)] # memory_size * len(s, a, r, s_)
		# temporary parameters
		self.learn_step_counter = 0
		self.memory_couter = 0
		# # # # # # # build mode
		if test:
			self.model        = load_model(f'DNNs/{model_name}.h5')
			self.model.summary()
		else:
			self.model        = self.build_ResNet_model() # model: evaluate Q value
		self.target_model = self.build_ResNet_model() # target_mode: target network

	def get_config(self):
		#config = super().get_config().copy()
		return {'state_size': self.state_size,
				'n_actions': self.n_actions,
				'next_state_shape': self.next_state_shape,
				'memory_size': self.memory_size,
				'replace_target_iter': self.replace_target_iter,
				'batch_size': self.batch_size,
				'learning_rate': self.learning_rate,
				'gamma': self.gamma,
				'epsilon': self.epsilon,
				'epsilon_min': self.epsilon_min,
				'epsilon_decay': self.epsilon_decay,
				'test': self.test}
		#return config
	

	def build_ResNet_model(self):

		inputs = layers.Input(shape=(self.state_size))
		x = layers.LSTM(units=64)(inputs)
		#x = layers.Conv1D(32, 8, activation="relu")(inputs)
		#x = layers.Conv1D(32, 4, activation="relu")(x)
		
		#Embedding(input_dim=self.state_size, output_dim=embed_dim)(inputs)
		
		x = layers.Dropout(0.1)(x)
		x = layers.Dense(32, activation="relu")(x)
		#outputs = layers.SimpleRNN(256)(x)
		x = layers.Dropout(0.1)(x)
		outputs = layers.Dense(self.n_actions, activation="softmax")(x)
		model = keras.Model(inputs=inputs, outputs=outputs)

		model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=self.learning_rate)) # loss="mse"

		return model

	def choose_action(self, state, t, check): # t = time slot
		state = np.concatenate(np.concatenate([ state[:] ]))
		state = state[np.newaxis]
		#print('***', np.shape(state))
		
		self.epsilon *= self.epsilon_decay
		
		"""
		if not self.test:
			if t == 5000 or check:
				self.epsilon -= .1
		"""
		
		#self.epsilon = self.epsilon_min + (self.epsilon_max-self.epsilon_min)*math.exp( -t/500 )
		
		self.epsilon = max(self.epsilon_min, self.epsilon)
		action_values = self.model.predict(state + np.ones_like(state))
		action = np.zeros(len(action_values[0]), dtype=int)
		
		if np.random.random() < self.epsilon or sum(action_values[0])==0:
			argu = np.random.randint(0, len(action_values[0]))
			action[argu] = 1
			return action, action_values[0][argu], action_values[0]
		
		argu = np.random.choice(self.n_actions, p=np.squeeze(action_values))
		action[argu] = 1
		return action, action_values[0][argu], action_values[0]

	def normalize(self, vector, verbose=0):
		#v = vector.copy()
		#if min(vector) < 0:
		#	for i in range(len(vector)):
		#		v[i] -= min(vector)
		v = vector/sum(vector)
		if verbose == 1:
			print(f'*** vector: {vector}, sum: {sum(vector)}, v: {v}, sum:{sum(v)}')
		return v

	def store_transition(self, s, a, r, s_): # s_: next_state
		if not hasattr(self, 'memory_couter'):
			self.memory_couter = 0
		state = np.concatenate(np.concatenate([ s[:] ]))
		#state = state[np.newaxis]
		state_ = np.concatenate(np.concatenate([ s_[:] ]))
		#state_ = state[np.newaxis]

		transition = ( state, a, r, state_ )
		index = self.memory_couter % self.memory_size
		self.memory[index] = transition
		self.memory_couter += 1

	def repalce_target_parameters(self):
		weights = self.model.get_weights()
		self.target_model.set_weights(weights)

	def pretrain_learn(self, state):
		state = state[np.newaxis, :]
		init_value = 0.5/(1-self.gamma)
		q_target = np.ones(3)*init_value
		q_target = q_target[np.newaxis, :]
		self.model.fit(state + np.ones_like(state), q_target, batch_size=1, epochs=1, verbose=0)

	def learn(self):
		# check to update target netowrk parameters
		if self.learn_step_counter % self.replace_target_iter == 0:
			self.repalce_target_parameters() # iterative target model
		self.learn_step_counter += 1

		# sample batch memory from all memory
		if self.memory_couter > self.memory_size:
			sample_index = np.random.choice(self.memory_size, size=self.batch_size)
		else:
			sample_index = np.random.choice(self.memory_couter, size=self.batch_size)
		batch_memory = []
		for x in sample_index:
			batch_memory.append(self.memory[x])

		# batch memory row: [s, a, r1, r2, s_] 
		# number of batch memory: batch size 
		# extract state, action, reward, reward2, next_state from batch memory
		state = []
		action = []
		reward = []
		next_state = []
		for x in batch_memory:
			state.append(x[0])
			action.append(x[1])
			reward.append(x[2])
			next_state.append(x[3])
		#print('action', action)
		#print('reward', reward)

		q_eval = []
		q_next = []
		#q_test = self.model.predict( (state + np.ones_like(state)) )
		for s, s_ in zip(state, next_state):
			q_eval.append( self.model.predict( (s + np.ones_like(s))[np.newaxis] )[0].tolist() ) # state
			q_next.append( self.target_model.predict( (s + np.ones_like(s))[np.newaxis] )[0].tolist() ) # next state
		q_target = np.array(q_eval).copy()

		batch_index = np.arange(self.batch_size, dtype=np.int32) # just a list from 0 to batch_size
		for i in batch_index:
			q_target[i][np.where(action[i]==1)[0][0]] =  reward[i] + self.gamma * q_target[i][np.argmax(q_next[i])] # np.max(q_next[i]) # * np.max(q_next, axis=1)

		#q_target = self.normalize(q_target, verbose=0)
		self.model.fit(state + np.ones_like(state), q_target, self.batch_size, epochs=1, verbose=0)

	def save_model(self, fn):
		self.model.save(fn)

