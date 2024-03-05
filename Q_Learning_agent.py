import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import random
import ujson as json  # Import json for dictionary serialization

class QLAgent:
    # here are some default parameters, you can use different ones
    def __init__(self, action_space, alpha=0.5, gamma=0.8, epsilon=.7, mini_epsilon=0.5, decay=0.999):
        self.action_space = action_space 
        self.alpha = alpha               # learning rate
        self.gamma = gamma               # discount factor  
        self.epsilon = epsilon           # exploit vs. explore probability
        self.mini_epsilon = mini_epsilon # threshold for stopping the decay
        self.decay = decay               # value to decay the epsilon over time
        if input("Create new table? (enter YES PLEASE): ") == "YES PLEASE":
            print("Creating new q-tables")
            self.q_table_navigation = self.initialize_table(-1,20,0,25,0.2)
            self.q_table_norms = self.initialize_table(-1,20,0,25,0.2)
        else:
            print("Using pre-existing q-tables")
            with open('qtable_navigation.json') as table_navi:
                self.q_table_navigation = json.load(table_navi)
            # print(self.q_table_navigation)
            with open('qtable_norms.json') as table_norms:
                self.q_table_norms = json.load(table_norms)
        # agent = QLAgent(action_space=5)
        
        # output_navigation = open("qtable_trial.txt", "w")
        # for k, v in self.q_table_navigation.items():
        #     output_navigation.writelines(f'{k} {v[0]} {v[1]} {v[2]} {v[3]} {v[4]} {v[5]}\n')
        # output_norms = open("qtable_trial.txt", "w")
        # for k, v in self.q_table_norms.items():
        #     output_norms.writelines(f'{k} {v[0]} {v[1]} {v[2]} {v[3]} {v[4]} {v[5]}\n')
        # self.qtable = pd.DataFrame(columns=[i for i in range(self.action_space)])  # generate the initial table
        # self.qtable = self.initialize_table(self.x_min, self.x_max, self.y_min, self.y_max, self.step_size)

    # exit_pos = [-0.8, 15.6] # The position of the exit in the environment from [-0.8, 15.6] in x, and y = 15.6
    # cart_pos_left = [1, 18.5] # The position of the cart in the environment from [1, 2] in x, and y = 18.5
    # cart_pos_right = [2, 18.5] 
    # x_min = 0
    # x_max = 19
    # y_min = 0
    # y_max = 24
    # granularity = 0.5
    # Encrypts coordinates and cart status into a state
    # Encrypts coordinates and cart status into a state
            
    def initialize_table(self, x_min, x_max, y_min, y_max, step_size):
        qtable = {}
        printEncode = 0
        i = x_min
        while (i < x_max):
            j = y_min
            while (j < y_max):
                qtable.update({self.encrypt(i, j, 0, 0, printEncode): [0,0,0,0,0,0,0,0]})
                qtable.update({self.encrypt(i, j, 1, 0, printEncode): [0,0,0,0,0,0,0,0]})
                qtable.update({self.encrypt(i, j, 0, 1, printEncode): [0,0,0,0,0,0,0,0]})
                qtable.update({self.encrypt(i, j, 1, 1, printEncode): [0,0,0,0,0,0,0,0]})
                j = j + step_size
            i = i + step_size
        return(qtable)
    
    def encrypt(self, x_coord, y_coord, has_cart, has_violation, printEncode = 1):
        x_coord = round(round(x_coord * 5.0) / 5, 2)
        y_coord = round(round(y_coord * 5.0) / 5, 2)
        if printEncode == 1:
            print("Encoding: "+str(x_coord)+", "+str(y_coord)+", cartStatus = "+str(has_cart)+" and violationStatus = "+str(has_violation))
            pass
        x_coord_encrypt = x_coord * 1000000
        y_coord_encrypt = y_coord * 1000
        if has_cart == 0:
            cartkey = 1
        else:
            cartkey = 0
        if has_violation == 1:
            violationkey = 50
        else:
            violationkey = 0

        # if has_violation == 0:
            
        learning_state = str(int(x_coord_encrypt + y_coord_encrypt + violationkey + cartkey))
        
        return learning_state

    # Decrypts the learning state into coordinates and cart status
    def decrypt(self, learning_state):
        # Finding X coordinate
        learning_state = int(learning_state)
        x_coord = round(round(learning_state/1000000, 2), 1)
        y_coord = round(((learning_state)-x_coord*1000000)/1000, 1)
        if learning_state % 2 == 0:
            has_cart = -1
        else:
            has_cart = 1
        if int(str(learning_state)[-2:]) > 49:
            has_violation = 1
        else:
            has_violation = 0
        return (x_coord, y_coord, has_cart, has_violation)

    def trans(self, state, printEncode = 1):
        # You should design a function to transform the huge state into a learnable state for the agent
        # It should be simple but also contains enough information for the agent to learn
        # state = recv_socket_data(sock_game)  # get observation from env
        # state = json.loads(state)
        x_coord = state['observation']['players'][0]['position'][0]
        y_coord = state['observation']['players'][0]['position'][1]
        has_cart = state['observation']['players'][0]['curr_cart']
        violation = state['violations']
        has_violation = 0
        if violation != "":
            has_violation = 1
            # print("WHOA")
        if printEncode == 1:
            learning_state = self.encrypt(x_coord, y_coord, has_cart, has_violation)
        else:
            learning_state = self.encrypt(x_coord, y_coord, has_cart, has_violation, printEncode = 0)
        return learning_state
    
    def learning(self, action, rwd, state, next_state):
        # implement the Q-learning function
        # print(state)
        # print(rwd)
        rwd_navigation = rwd[0]
        rwd_norms = rwd[1]
        # print(next_state)
        # print(self.q_table_navigation[next_state])
        # print(max(self.q_table_navigation[next_state]))
        # new_value_navigation = old_value_navigation + self.alpha * (rwd_navigation + self.gamma * max(self.q_table_navigation[next_state]) - old_value_navigation)
        old_value_navigation = self.q_table_navigation[state][action]
        new_value_navigation = old_value_navigation + self.alpha * (rwd_navigation + self.gamma * max(self.q_table_navigation[next_state]) - old_value_navigation)
        self.q_table_navigation[state][action] = new_value_navigation

        old_value_norms = self.q_table_norms[state][action]
        new_value_norms = old_value_norms + self.alpha * (rwd_norms + self.gamma * max(self.q_table_norms[next_state]) - old_value_norms)
        self.q_table_norms[state][action] = new_value_norms
        pass

    def choose_action(self, state):
        choices = [0,1,2,3,4,5,6]
        
        # Combine the state lists for both Q-tables
        combined = []
        i = 0
        been_before = False

        while i < 8:
            temp_sum = self.q_table_navigation[state][i] + self.q_table_norms[state][i]
            combined.append(temp_sum)
            if abs(temp_sum) > 0:
                been_before = True
            i = i + 1
            
        # print(combined)
        
        # If agent has never been here before, pick a random action index
        if been_before == False:
            action_index = random.choice(choices)
            # print("Been Before: NO | Random: YES | Choice: "+str(action_index))
        else:
            # If agent has been here before, roll epsilon dice
            # If random, choose random action and decay epsilon
            if random.uniform(0,1) <= self.epsilon:
                if self.epsilon >= self.mini_epsilon:
                    self.epsilon = self.epsilon * self.decay
                action_index = random.choice(choices)
                # print("Been Before: YES | Random: YES | Choice: "+str(action_index))
            else:
                prob = F.softmax(torch.tensor(combined[:-1]), dim=0).detach().numpy()
                action_index = np.random.choice(np.flatnonzero(prob == prob.max()))
                # print("Been Before: YES | Random: NO | Choice: "+str(action_index))
        # print("Epsilon: "+str(self.epsilon))
            
            
            
            # # If not random, find highest value and choose that index
            #     temp_max = combined[0]
            #     temp_index = 0
            #     tied_max_indices = []
            #     i = 0
            #     while i < len(combined)-1:
            #         if combined[i] > temp_max:
            #             temp_max = combined[i]
            #             temp_index = i
            #         i = i + 1
            #     i = 0
            #     while i < len(combined)-1:
            #         if combined[i] == temp_max:
            #             tied_max_indices.append(i)
            #         i = i + 1
            #     # print("ZERO INDEX "+str(zero_indices))
            #     # If the max is tied, then choose randomly between indexes with the max
            #     if len(tied_max_indices) > 1:
            #         action_index = random.choice(tied_max_indices)
            #         print(tied_max_indices)
            #         print("Been Before: YES | Random: TIED | Choice: "+str(action_index))
            #     # Chooses temp_index if there is a definite winner
            #     else:
            #         action_index = temp_index
            #         print("Been Before: YES | Random: NO | Choice: "+str(action_index))

        return action_index

        # If agent has been here, roll epsilon dice (and decay)


        # If never been at this state before
        # i = 0
        # maxabs = 0
        # while i < len(self.q_table_navigation[state]):
        #     if abs(self.q_table_navigation[state][i]) > maxabs:
        #         maxabs = abs(self.q_table_navigation[state][i])
        #     i += 1
        # if maxabs == 0:
        #     # Never been there before
        #     index = random.choice(choices)
        #     print("Random Choice: YES")
        # # If random from epsilon
        # elif random.uniform(0,1) < self.epsilon:
        #     index = random.choice(choices)
        #     print("Random Choice: YES")
        #     # Decay over time
        #     if self.epsilon >= self.mini_epsilon:
        #         self.epsilon = self.epsilon*self.decay
        # # Choosing from QTable
        # else:
        #     # Access Q-Table and find index with highest value, then chooses that index for action
        #     options = []
        #     i = 0
        #     while (i < len(self.q_table_navigation[state])):
        #         option = self.q_table_navigation[state][i] + self.q_table_norms[state][i]
        #         options.append(option)
        #         i = i + 1
        #     max_element = options[0]
        #     index = 0
        #     max_index = 0
        #     print(options)
        #     # for i in range (0,len(self.q_table[state])):
        #     while (index < len(options)-1):
        #         if options[index] > max_element:
        #             max_element = options[index]
        #             max_index = index
        #         index = index + 1
        #     print(max_element)
        #     if max_index == 7 or max_element == 0:
        #         max_index = random.choice([0,1,2,3,4,5,6])
        #     index = max_index
        #     # print("Random Choice: NO")
        #     # print("Random Choice: NO | optionsList = "+str(options)+" | chosen = "+str(index))
        #     print("Random Choice: NO | chosen = "+str(index))
        # print("Epsilon: "+str(self.epsilon))
        # return index
 
# agent = QLAgent(action_space=5)
# q_table = agent.initialize_table(0,19,0,24,0.5)

