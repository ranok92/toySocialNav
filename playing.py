"""
Once a model is learned, use this to play it. that is run/exploit a policy to get the feature expectations of the policy
"""
import sys
from flat_game import ballgamepyg as BE
import numpy as np
from nn import Neural_net as NN
import math
import time
import csv


def play(weights,saved_model = None,display = False , nnParams = None  ,featsize = None):
    #given the weights create the reward function
    #using that reward function and the model agent play the game
    #and return the feature expectation of the gameplay
    FEATURESIZE = featsize
    #each playing episode lasts for 2000 time steps
    #or if the game ends. Which ever comes earlier.s
    TIMESTEPS = 1000
    NO_OF_TRIALS = 10
    GAMMA = .99
    FeatureExpectation = np.zeros((FEATURESIZE,)) #44 is the size of the sensor_reading array
    board = BE.createBoardIRL(sensor_size= FEATURESIZE ,weights= weights, saved_model= saved_model,display = display , hidden_layers= nnParams)
    for i in range(NO_OF_TRIALS):
        board.reset()
        FEperTrial = np.zeros((FEATURESIZE,))

        for j in range(TIMESTEPS):
            if board.display:
                board.render()
            #action = cb.take_action_from_user()
            action = board.gen_action_from_agent()
            action = board.agent_action_to_WorldAction(action)
            #print action
            #print cb.sensor_  readings
            state ,reward ,done , _ = board.step(action)
            FEperTrial = np.add(FEperTrial,math.pow(GAMMA,j)*board.sensor_readings)
            if done:
                break
        FeatureExpectation = np.add(FeatureExpectation , FEperTrial)
        #print "for trial {0} : ".format(i),FEperTrial
    FeatureExpectation = np.divide(FeatureExpectation,NO_OF_TRIALS)
    FeatureExpectation = np.divide(FeatureExpectation,500) #500 because size of the current environment is 500 x 500
    board.quit_game()

    return FeatureExpectation

def test_model(weights, saved_model_path, k , nnParams = None , sensor_size = None):
    TIMESTEPS =50
    NO_OF_TRIALS = 100
    GAMMA = .99
    board = BE.createBoardIRL(sensor_size= sensor_size, weights = weights , display= True, saved_model = saved_model_path , hidden_layers= nnParams)
    acc_reward = []
    for i in range(NO_OF_TRIALS):
        board.reset()

        for j in range(TIMESTEPS):
            if board.display:
                board.render()

            action = board.gen_action_from_agent()
            print 'action',action
            action = board.agent_action_to_WorldAction(action)

            state, reward ,done, _ = board.step(action)
            print reward
            if done:
                break
        #store the accumulated reward in a log file
        acc_reward.append(['Run '+str(i) ,board.total_reward_accumulated])


    board.quit_game()
    with open("results/testRuns/"+'Iteration_no_'+str(k)+".csv",'w') as fl:
        wr = csv.writer(fl)
        for res in acc_reward:
            wr.writerow(res)

def play_User(NO_OF_TRIALS,nn_params,FEATSIZE):
    FEATURESIZE = FEATSIZE
    FeatureExpectation = np.zeros([FEATURESIZE,1])
    TIMESTEPS = 1000
    weights = np.random.rand(FEATURESIZE)
    #print 'WEIGHTSshape', weights.shape
    board = BE.createBoardIRL(sensor_size = FEATSIZE, display=True, weights= weights , hidden_layers= nn_params)
    #print "Board created...",board
    #board.display= True
    GAMMA = .99
    for i in range(NO_OF_TRIALS):
        board.reset()
        FEperTrial = np.zeros([FEATURESIZE,1])
        for t in range(TIMESTEPS):
            if board.display:
                board.render()

            action = board.take_action_from_userKeyboard()
            action = board.agent_action_to_WorldActionSimplified(action)
            #print action
            state, reward, done, _ = board.step(action)
            print board.sensor_readings.shape
            FEperTrial = np.add(FEperTrial,np.multiply(math.pow(GAMMA,t),np.expand_dims(board.sensor_readings,1)))
            print 'FEPERTRAIl', FEperTrial.shape
            if done:
                break
        FeatureExpectation = np.add(FeatureExpectation , FEperTrial)
        #print "for trial {0} : ".format(i),FEperTrial
        FeatureExpectation = np.divide(FeatureExpectation,NO_OF_TRIALS)


    print 'FEShape',FeatureExpectation.shape
    np.savetxt('newFTFile2_1feat.txt',FeatureExpectation)
    board.quit_game()

    return FeatureExpectation




if __name__ == '__main__':


    FEATSIZE = 5

    #weights = np.genfromtxt('EXPERTFEWeights.csv')
    nn_param = [32,32]

    #run this part to test a model
    #weights = [-.4, 0, 0 , 0 ]
    #weights = np.asarray(weights)
    #model_path = '/home/abhisek/Study/Robotics/toySocialNav/saved-models_train_from_main/evaluatedPoliciesTest/2018-12-05/23:02:21.235622/111-32-32-32-100-50000-6000.h5'
    #test_model(weights , model_path, 10, nnParams=nn_param , sensor_size= FEATSIZE)


    #run this part to collect expertfeature expectation
    play_User(50,nn_param,FEATSIZE)
