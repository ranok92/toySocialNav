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
'''
NUM_STATES = 8
GAMMA = 0.9


def play(model, weights):

    car_distance = 0
    game = BE.createBoard()



    _, state, __ = game_state.frame_step((2))

    featureExpectations = np.zeros(len(weights))

    # Move.
    #time.sleep(15)
    while True:
        car_distance += 1

        # Choose action.
        action = (np.argmax(model.predict(state, batch_size=1)))
        #print ("Action ", action)

        # Take action.
        immediateReward , state, readings = game_state.frame_step(action)
        #print ("immeditate reward:: ", immediateReward)
        #print ("readings :: ", readings)
        #start recording feature expectations only after 100 frames
        if car_distance > 100:
            featureExpectations += (GAMMA**(car_distance-101))*np.array(readings)
        #print ("Feature Expectations :: ", featureExpectations)
        # Tell us something.
        if car_distance % 2000 == 0:
            print("Current distance: %d frames." % car_distance)
            break


    return featureExpectations

if __name__ == "__main__": # ignore
    BEHAVIOR = sys.argv[1]
    ITERATION = sys.argv[2]
    FRAME = sys.argv[3]
    saved_model = 'saved-models_'+BEHAVIOR+'/evaluatedPolicies/'+str(ITERATION)+'-164-150-100-50000-'+str(FRAME)+'.h5'
    weights = [-0.79380502 , 0.00704546 , 0.50866139 , 0.29466834, -0.07636144 , 0.09153848 ,-0.02632325 ,-0.09672041]
    model = NN(NUM_STATES, [164, 150], saved_model)
    print (play(model, weights))
'''


def play(weights,saved_model = None,display = False):
    #given the weights create the reward function
    #using that reward function and the model agent play the game
    #and return the feature expectation of the gameplay

    #each playing episode lasts for 2000 time steps
    #or if the game ends. Which ever comes earlier.
    TIMESTEPS = 500
    NO_OF_TRIALS = 100
    GAMMA = .9
    FeatureExpectation = np.zeros(44) #44 is the size of the sensor_reading array
    board = BE.createBoardIRL(weights= weights, saved_model= saved_model,display = display)
    for i in range(NO_OF_TRIALS):
        board.reset()
        FEperTrial = np.zeros(44)

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
        print "for trial {0} : ".format(i),FEperTrial
    FeatureExpectation = np.divide(FeatureExpectation,NO_OF_TRIALS)

    board.quit_game()

    return FeatureExpectation

def test_model(weights, saved_model_path,k):
    TIMESTEPS = 500
    NO_OF_TRIALS = 100
    GAMMA = .9
    board = BE.createBoardIRL(weights = weights , saved_model = saved_model_path)
    acc_reward = []
    for i in range(NO_OF_TRIALS):
        board.reset()
        for j in range(TIMESTEPS):
            if board.display:
                board.render()

            action = board.gen_action_from_agent()
            action = board.agent_action_to_WorldAction(action)

            state, reward ,done, _ = board.step(action)

            if done:
                break
        #store the accumulated reward in a log file
        acc_reward.append(['Run '+str(i) ,board.total_reward_accumulated])


    board.quit_game()
    with open("results/testRuns/"+'Iteration_no_'+str(k)+".csv",'w') as fl:
        wr = csv.writer(fl)
        for res in acc_reward:
            wr.writerow(res)






if __name__ == '__main__':

    weights = [-0.00568885,  0.02006721,  0.00645485,  0.06885036, -0.06536262,  0.02495588,
 -0.05763496,  0.26426469,  0.00195435,  0.17015842, -0.06406653, -0.06718497,
 -0.06365476 ,-0.05691974, -0.06323244, -0.06245765, -0.06452598, -0.06458144,
 -0.07061446, -0.06799019, -0.07375392, -0.07260787, -0.06615446, -0.06396543,
 -0.05943623, -0.05779849, -0.05775174, -0.06156051, -0.06386689, -0.06426792,
 -0.06150341, -0.05391958, -0.05994149,  0.18965812, -0.06019738,  0.06186743,
 -0.02512689,  0.20833798,  0.00512722, -0.01422591,  0.21285729,  0.59172853,
 -0.55742996,  0.02200511]
    weights = np.asarray(weights)

    model_path = '/home/abhisek/Study/Robotics/toyCarIRL/saved-models_red/evaluatedPolicies/81-164-150-100-50000-1000.h5'

    play(weights = weights,saved_model= model_path ,display = True)
