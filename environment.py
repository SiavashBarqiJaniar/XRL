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
from numpy import *

class ENVIRONMENT(object):
    """docstring for ENVIRONMENT"""
    def __init__(self,
				 state_shape = (20, 9),
                 action_size = 9,
                 next_state_shape = 10,
				 ):
        super(ENVIRONMENT, self).__init__()
        self.state_shape = state_shape
        self.action_space = [f'C{i}' for i in range(action_size)] # w: wait t: transmit
        self.n_actions = action_size
        self.next_state_shape = next_state_shape
        self.state = self.reset()
    
    def reset(self):
        init_state = np.zeros(self.n_actions, int)
        return init_state
    
    def step(self, action, lmbda, jammer_action):
        n = len(action)
        mu = 1
        
        random_jammer_action = np.random.randint(0, n) # uniform?

        #print('actions:', np.where(action==1)[0][0], '&', np.where(jammer_action==-1)[0])
        if (np.where(action==1)[0][0] == np.where(jammer_action==-1)[0]).any():
        # Collision?:
            #print('COLLISION')
            mu = 0
            self.state = jammer_action.copy()
            self.state[np.where(action==1)[0][0]] = -2
        else:
            self.state = action + jammer_action
        #self.state[random_jammer_action] = -1
        #==============


        """
        reward = agent_reward
		"""
        return mu, self.state

