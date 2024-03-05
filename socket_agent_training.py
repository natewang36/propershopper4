#Author Hang Yu

import ujson as json
import random
import socket

import gymnasium as gym
from env import SupermarketEnv
from utils import recv_socket_data

from Q_Learning_agent import QLAgent  # Make sure to import your QLAgent class
import pickle
import pandas as pd

destination_pos = [1, 17.5]

def distance_to_destination(learning_state):
    # info[0] is x, info[1] is y, info[2] is cart
    info = agent.decrypt(learning_state)
    agent_position = (info[0], info[1])
    # print(agent_position)
    # if info[0] > 1.5
    #     distance = euclidean_distance(agent_position, cart_pos_right)
    # else:
    #     distance = euclidean_distance(agent_position, cart_pos_left)
    distance = euclidean_distance(agent_position, destination_pos)
    return distance

def euclidean_distance(pos1, pos2):
# Calculate Euclidean distance between two points
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

def has_cart(learning_state):
    # info[0] is x, info[1] is y, info[2] is cart
    info = agent.decrypt(learning_state)
    if info[2] == 1:
        return 1
    else:
        return 0
    
def has_violation(learning_state):
    info = agent.decrypt(learning_state)
    if info [3] == 1:
        return 1
    else:
        return 0

def calculate_reward(state, next_state):
    # design your own reward function here
    # You should design a function to calculate the reward for the agent to guide the agent to do the desired task
    rwd_navigation = 0
    rwd_norms = 0
    step_penalty = -0.1  # Small penalty for each step
    rwd_navigation += step_penalty
    x_coord = agent.decrypt(state)[0]
    y_coord = agent.decrypt(state)[1]
    x_coord_next = agent.decrypt(next_state)[0]
    y_coord_next = agent.decrypt(next_state)[1]

    # To discourage resetting
    # if abs(distance_to_destination(state) - distance_to_destination(next_state)) > 0.3:
    #     rwd_navigation += -100000000

    # To encourage getting closer while not at destination
    if distance_to_destination(state) >= 0.2:
        print("On the way")
        # print(str(x_coord)+" - "+str(x_coord_next)+" | "+str(y_coord)+" - "+str(y_coord_next))
        # If moving
        
        # if abs(x_coord - x_coord_next) > 0.1 or abs(y_coord - y_coord_next) > 0.1:
        #     rwd_navigation += (distance_to_destination(state) - distance_to_destination(next_state)) * 10
        # else:
        #     rwd_navigation -= distance_to_destination(state)
        
        if x_coord - x_coord_next > 0.1 or y_coord - y_coord_next > 0.1:
            rwd_navigation += (distance_to_destination(state) - distance_to_destination(next_state)) * 10
        elif x_coord_next - x_coord > 0.1 or y_coord_next - y_coord > 0.1:
            rwd_navigation == 0
        else:
            rwd_navigation -= distance_to_destination(state)

    # To reward staying at destination
    else:
        print("At the destination!")
        rwd_navigation += 100

    
    if has_violation(next_state) == 1:
        rwd_norms += -100

    print("Distance to Destination: "+str(distance_to_destination(next_state)))

    reward = [rwd_navigation, rwd_norms]
    # print(reward)
    return reward

if __name__ == "__main__":
    

    action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT', 'RESET'] #TOGGLE_CART
    # Initialize Q-learning agent
    action_space = len(action_commands) - 1   # Assuming your action space size is equal to the number of action commands
    agent = QLAgent(action_space)

####################
    #Once you have your agent trained, or you want to continue training from a previous training session, you can load the qtable from a json file
    # agent.q_table_navigation = pd.read_json('qtable_navigation.json')
    # agent.q_table_norms = pd.read_json('qtable_norms.json')
####################
    
    
    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = 9000
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    training_time = 1000
    episode_length = 1000
    success_counter = 0
    for i in range(training_time):
        sock_game.send(str.encode("0 RESET"))  # reset the game
        json_state = recv_socket_data(sock_game)
        json_state = json.loads(json_state)
        state = agent.trans(json_state, printEncode=0)
        cnt = 0
        success = False
        # print(str(i)+" -----------------")
        while not json_state['gameOver']:
            cnt += 1
            print("Success Counter: "+str(success_counter)+" / "+str(i+1))
            print("Episode: "+str(i+1)+" Move: "+str(cnt))
            # Choose an action based on the current state
            print("------------------------------------------")
            action_index = agent.choose_action(state)
            action = "0 " + action_commands[action_index]

            # print("Sending action: ", action)
            # sock_game.send(str.encode(action))  # send action to env
            
            # json_next_state = recv_socket_data(sock_game)  # get observation from env
            # json_next_state = json.loads(json_next_state)
            sock_game.send(str.encode(action))  # send action to env
            
            json_next_state = recv_socket_data(sock_game)  # get observation from env
            json_next_state = json.loads(json_next_state)
            sock_game.send(str.encode(action))  # send action to env
            
            json_next_state = recv_socket_data(sock_game)  # get observation from env
            json_next_state = json.loads(json_next_state)
            next_state = agent.trans(json_next_state, 1)
            # sock_game.send(str.encode(action))  # send action to env
            if distance_to_destination(state) < 0.2 and success == False:
                success = True
                success_counter += 1
            
            # json_next_state = recv_socket_data(sock_game)  # get observation from env
            # json_next_state = json.loads(json_next_state)
            # next_state = agent.trans(json_next_state)
            # Define the reward based on the state and next_state
            reward = calculate_reward(state, next_state)  # You need to define this function
            print("Action: "+str(action_commands[action_index])+" | Reward: "+str(reward)+" | State: "+str(next_state))
            print("------------------------------------------")
            print("   ")
            print("   ")

            if cnt < episode_length:
                # Update Q-table
                agent.learning(action_index, reward, state, next_state)

            # Update state
            state = next_state
            # agent.q_table.to_json('qtable.json')
            with open("qtable_navigation.json", "w") as outfile_navi: 
                json.dump(agent.q_table_navigation, outfile_navi)
            with open("qtable_norms.json", "w") as outfile_norms: 
                json.dump(agent.q_table_norms, outfile_norms)

            if cnt > episode_length:
                break
        # Additional code for end of episode if needed

    # Close socket connection
    sock_game.close()

