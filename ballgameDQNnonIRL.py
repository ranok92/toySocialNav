import gym
import math
import os
import os.path
import datetime


import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import featureExtractor
import torch
import torch.nn as NN
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import ballenv_pygame as BE
from nn import Neural_net as neural_net

'''
Short description of the file :

This file was used to test the DQN method for the ballgame environment.
This is a standalone file and does not plugin with the IRL pipeline. 
Although it takes the architecture from the nn.py file.

** Might be a duplicate of the file learningBlock.py 

'''





steps_done = 0
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

steps_done = 0

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



#changes to make for different environments search for 'change here'

#get state from classical gymball ENvironment
def block_to_arrpos(x,y):

    pos = 12+(5*y)+x
    return int(pos)



#takes the screen pixel information
#takes the state and converts that into an array
#size of array depends on the number of obstacles present in the environment
def get_state_BallEnvFullState(state):
    ln = len(state)
    #print 'LNN',ln
    #print 'The original state :',state
    conv_state = np.zeros((ln-1)*2+1)
    conv_state[0] = state[0][0]
    conv_state[1] = state[0][1]
    conv_state[2] = state[1][0]
    conv_state[3] = state[1][1]
    conv_state[4] = state[2]
    counter = 5
    for i in range(ln-3):
        obs = state[i+3]
        #print obs
        conv_state[counter] = obs[0]
        counter+=1
        conv_state[counter] = obs[1]
        counter+=1

    conv_state = np.divide(conv_state,200)
    state = torch.from_numpy(conv_state).to(device)
    state = state.type(torch.cuda.FloatTensor)
    state = state.unsqueeze(0)
    #print 'The state after conversion :', state
    return state




def get_state_BallEnv(state):

#state is a list of info where 1st position holds the position of the
#agent, 2nd the position of the goal , 3rd the distance after that,
#the positions of the obstacles in the world
    #print(state)
    ref_state = np.zeros(29)
    #print(ref_state.shape)
    ref_state[12+4] = 1
    agent_pos = state[0]
    goal_pos = state[1]
    diff_x = goal_pos[0] - agent_pos[0]
    diff_y = goal_pos[1] - agent_pos[1]
    if diff_x >= 0 and diff_y >= 0:
        ref_state[1] = 1
    elif diff_x < 0  and diff_y >= 0:
        ref_state[0] = 1
    elif diff_x < 0 and diff_y < 0:
        ref_state[3] = 1
    else:
        ref_state[2] = 1

    for i in range(3,len(state)):
        x_dist = agent_pos[0] - state[i][0]
        y_dist = agent_pos[1] - state[i][1]
        x_block = y_block = 0
        #print("XY",x_dist,y_dist)
        if x_dist!=0 and y_dist!=0:
            if x_dist < 0:
                x_block = ((x_dist/abs(x_dist))*(x_dist-10)//20)+1
            else:
                x_block = (x_dist/abs(x_dist))*(x_dist-10)//20
            if y_dist < 0:
                y_block = ((y_dist/abs(y_dist))*(y_dist-10)//20)+1
            else:
                y_block = (y_dist/abs(y_dist))*(y_dist-10)//20
            #print("BLOCK",x_block,y_block)
        if (abs(x_block)<3 and abs(y_block)<3): #considering a matrix of 5x5 with the agent being present at the center
        #print(x_block,y_block)
            pos = block_to_arrpos(x_block, y_block) #pos starts from 0
            #print("POS",pos)
            ref_state[4+pos] += 1
    #state is as follows:
        #first - tuple agent position
        #second -
    state = torch.from_numpy(ref_state).to(device)
    state = state.type(torch.cuda.FloatTensor)
    state = state.unsqueeze(0)

    return state


#get state from cartpole environment
def get_state_CartPole(statelist):

    state = np.asarray(statelist)
    state = torch.from_numpy(state).to(device)
    state = state.type(torch.cuda.FloatTensor)
    state = state.unsqueeze(0)
    return state


#state for frozenLake is just an integer
def get_state_FrozenLake(state):

    newstate = np.zeros(16)
    newstate[state] = 1
    #state = np.asarray([state])
    state = torch.from_numpy(newstate).to(device)
    state = state.type(torch.cuda.FloatTensor)
    state = state.unsqueeze(0)

    return state

#state is a numpy array of size (2,)
def get_state_MountainCar(state):

    state = torch.from_numpy(state)
    state = state.type(torch.cuda.FloatTensor)
    state = state.unsqueeze(0)
    return state


def IRL_helper(weights, path, trainFrames, i,FEATSIZE , nn_param):
    nn_param = nn_param
    params = {
        "batchSize": 100,
        "buffer": 50000,
        "nn": nn_param
    }
    ####model = neural_net(NUM_INPUT, nn_param , num_actions=4)

    #train_net() - the first parameter used to be the model, but now as because the model
    # is being created inside the game environment, pass in either the path to a '/home/abhisek/Study/Robotics/toySocialNav/saved-models_red/eval'/home/abhisek/Study/Robotics/toySocialNav/saved-models_red/eval'/home/abhisek/Study/Robotics/toySocialNav/saved-models_red/eval'/home/abhisek/Study/Robotics/toySocialNav/saved-models_red/evaluatedPolicies/17-512-1024-100-50000-10011.h5'uatedPolicies/17-512-1024-100-50000-10011.h5'uatedPolicies/17-512-1024-100-50000-10011.h5'uatedPolicies/17-512-1024-100-50000-10011.h5'saved model or None
    #train_net method returns the path in which the model dictionary is saved
    return train_model_main(None, params, weights, path, trainFrames, i,FEATSIZE)


def params_to_filename(params):
    return str(params['nn'][0]) + '-' + str(params['nn'][1])+ '-'+ str(params['nn'][1]) + '-' + \
            str(params['batchSize']) + '-' + str(params['buffer'])



def optimize_model(memory , policy_net, target_net, GAMMA , BATCH_SIZE , optimizer):

    if len(memory) < 1000:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))


    #print 'The type for batch.nextstate :', type(batch.next_state[0])
    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)


    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state, dim=0)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    #print 'State batch :',batch.state[0].shape
    #print 'bathsjahdfjha', state_batch
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    #print loss.item()
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def agent_action_to_WorldActionSimplified(action):
    if action==0: #move front
        return np.asarray([0,-5])
    if action==1: #move right
        return np.asarray([5,0])
    if action==2: #move down
        return np.asarray([0,5])
    if action==3: #move left
        return np.asarray([-5,0])



def select_action(state , policy_net , EPS_END , EPS_START , EPS_DECAY):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            #print 'before calling :',state.shape
            qvals = policy_net(state)
            #qvals = qvals.unsqueeze(0)
            #print 'Qval info :', qvals.shape
            #print qvals
            #print qvals.max(0)
            acts = qvals.max(1)[1]
            #print 'Qvals',qvals.shape
            #print "ACTS",acts
            trans = acts.view(1,1)
            return trans
    else:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)


#this should return the filename where the latest model is stored

#training_parameters - dictionary storing batchsize, hiddenlayer size , memory_size
def train_model_main(model_path , training_parameters , weights , path, trainIterations, i , featureSize ,actionSize):

    #*****************for storing the saved model************
    print "start Training . . ."
    filename = params_to_filename(training_parameters)
    curDay = str(datetime.datetime.now().date())
    curtime = str(datetime.datetime.now().time())
    basePath = 'saved-models_'+ path +'/evaluatedPoliciesTest/'
    subPath = curDay + '/' + curtime + '/'
    curDir = basePath + subPath
    os.makedirs(curDir)
    if os.path.exists(curDir):
        print "YES"

    #########################################################



    #change here
    #to change the environment

    #game = BE.createBoardIRL(sensor_size=featureSize,display= False, saved_model=model_path , weights = weights , hidden_layers= training_parameters['nn'])

    game = BE.createBoard(display= True,static_obstacles=20)

    policy_net = neural_net(featureSize , training_parameters['nn'],actionSize)
    policy_net.cuda()
    #create the target network
    targetNetwork = neural_net(featureSize , training_parameters['nn'],actionSize)
    stepcounter = 0 #keeps track of how many steps the learning network has taken ahead of the target network
    UPDATETARGET = 50 #no of times after which the target network needs to be updated
    targetNetwork.load_state_dict(policy_net.state_dict())
    targetNetwork.eval()
    targetNetwork.cuda()

    criterion = NN.SmoothL1Loss() #huber loss


    #criterion = NN.MSELoss()
    # Get initial state by doing nothing and getting the state.
    #_, state, temp1 = game_state.frame_step((2))

    #reset returns an array

    state = game.reset()

    #convert the state to tensor
    state = get_state_BallEnvFullState(state)

    BATCH_SIZE = training_parameters['batchSize']
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 5000
    TARGET_UPDATE = 1000
    #change here
    #############
    num_sensors = featureSize
    params = training_parameters['nn']
    num_actions = actionSize
    #############
    episode_durations = []
    rewardList = []
    runList = []
    #policy_net = DQN().to(device)
    #target_net = DQN().to(device)
    #target_net.load_state_dict(policy_net.state_dict())
    #target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(training_parameters['buffer'])

    fig = plt.figure(1)
    plt.clf()
    steps_done = 0
    num_episodes = trainIterations
    steps_per_episode = 500
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        state = game.resetFixedstate()
        #taking features from the game

        #stateNN =  game.sensor_readings
        stateNN = get_state_BallEnvFullState(state)
        print 'Starting episode :', i_episode
        for t in range(steps_per_episode):
            #get_state_MountainCar
            # Select and perform an action
            #game.render()
            actionIndex = select_action(stateNN, policy_net , EPS_END , EPS_START , EPS_DECAY)

            #print 'ActionIndex info', actionIndex

            action = agent_action_to_WorldActionSimplified(actionIndex)


            next_state, reward, done, _ = game.step(action)

            #print 'The reward obtained form the environment :',reward

            reward = torch.tensor([reward], device=device)

            reward = reward.type(torch.cuda.FloatTensor)

            # Observe new state
            #last_screen = current_screen
            #current_screen = get_screen()

            #change here
            if not done:
                next_stateNN = get_state_BallEnvFullState(next_state)
                #next_stateNN = game.sensor_readings
            else:
                next_stateNN = None

            if done:
                print 'Reward for this game :',game.total_reward_accumulated
                rewardList.append(game.total_reward_accumulated)
                runList.append(i_episode)
                plt.plot(runList, rewardList)
                plt.draw()
                plt.pause(.0001)
                fig.show()
                #next_state = current_state

            # Store the transition in memory
            #print state.shape
            memory.push(stateNN, actionIndex, next_stateNN, reward)

            # Move to the next state
            stateNN = next_stateNN

            # Perform one step of the optimization (on the target network)
            optimize_model(memory , policy_net, targetNetwork, GAMMA , BATCH_SIZE , optimizer)

            #change here (for mountain car only)

            if done:
                #episode_durations.append(t)
                #plot_durations()
                break


        # Update the target network
        if i_episode % TARGET_UPDATE == 0:
            targetNetwork.load_state_dict(policy_net.state_dict())

        if i_episode %50 == 0 and i_episode != 0:
            global steps_done
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                            math.exp(-1. * steps_done / EPS_DECAY)
            print 'Current eps_threshold :', eps_threshold
            print 'Steps DOne :', steps_done
            testModel(policy_net ,game, 5 , featureSize , True , weights , training_parameters , i_episode)

        if i_episode % 500 == 0 and i_episode != 0:

            torch.save(policy_net.state_dict(),'saved-models_'+ path +'/evaluatedPoliciesTest/'+subPath+str(i)+'-'+ filename + '-' + str(i_episode) + '.h5', )
            savedFilename = 'saved-models_'+ path +'/evaluatedPoliciesTest/'+subPath+str(i)+'-'+ filename + '-' + str(i_episode) + '.h5'
            with open('results/model_paths.txt','w') as ff:
                ff.write('saved-models_'+ path +'/evaluatedPoliciesTest/'+subPath+str(i)+'-'+ filename + '-' + str(i_episode) + '.h5\n')
                ff.close()
            print("Saving model %s - %d" % (filename, t))


    print('Complete')

    testModel(policy_net , game ,100 , featureSize , True ,weights , training_parameters , i_episode)
    #game.render()
    #env.close()
    plt.ioff()
    plt.show()

    return 0


def testModel( model , game ,no_of_trials, featureSize,display, weights, training_parameters , startEpoch):

    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    maxSteps = 100
    #game = BE.createBoard(display=True,static_obstacles=5)

    for i in range(no_of_trials):

        state = game.resetFixedstate()
        state = get_state_BallEnvFullState(state)
        #state = game.sensor_readings
        print 'The type of the state:',type(state)
        done = False
        rewardList = []
        game.render()

        t =  0 #steps in an episode
        while not done:

            actionIndex = select_action(state ,model , EPS_END , EPS_START , EPS_DECAY)

            action = agent_action_to_WorldActionSimplified(actionIndex)

            next_state , reward , done , _ = game.step(action)

            #next_state here is the next_state, we need the sensor readings
            game.render()
            rewardList.append(reward)
            #when game.step() is called, the value of the variable self.sensor_readings is internally
            #updated with the value obtained from the new state obtained from the action
            state  = get_state_BallEnvFullState(next_state)
            #state = game.sensor_readings
            t+=1
            if t > maxSteps:

                done = True

            if done:
                print 'The reward for this run :', game.total_reward_accumulated
                plt.figure()
                plt.plot(rewardList)
                pltfile = 'saved_plots/'+'smallexperiment/'+'featsize-'+str(featureSize)+str(+startEpoch+i)+'-'+'epoch'
                plt.savefig(pltfile+'.png')
                #playing.test_model(weights, )
                plt.close()

    #game.quit_game()


if __name__ == "__main__":
    weights = np.array([[-.4, 0, 0 , 0 ,0]])

    print 'The shape of the weight array', weights.shape
    path = 'trainBlock'
    FEATSIZE = 45
    ActionSize = 4
    nn_param = [1000,500]
    params = {
    "batchSize": 100,
    "buffer": 50000,
    "nn": nn_param }
    train_model_main(None, params, weights, path, 50000,111,FEATSIZE,ActionSize)
