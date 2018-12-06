# IRL algorith developed for the toy car obstacle avoidance problem for testing.
import numpy as np
import logging
import torch
import scipy
#import ballenv_pygame as BE
from playing import play   #get the RL Test agent, gives out feature expectations after 2000 frames
from playing import test_model
from playing import play_User
from nn import Neural_net as neural_net #construct the nn and send to playing
from cvxopt import matrix 
from cvxopt import solvers #convex optimization library
#from flat_game import carmunk # get the environment
from learning import IRL_helper # get the Reinforcement learner
from learning import get_saved_modelFilename
import os
import math

import matplotlib.pyplot as plt
import matplotlib.animation as animation
NUM_STATES = 8 
BEHAVIOR = 'red' # yellow/brown/red/bumping
FRAMES = 10000 # number of RL training frames per iteration of IRL
ENVIRONMENTSIZE = 100
class irlAgent:
    def __init__(self, randomFE, expertFE, epsilon, num_states, num_frames, behavior , featsize):
        self.randomPolicy = randomFE
        #self.expertTrajFile = expertTrajFile #directory that contains the expert trajetories

        # The trajectories are of the of the format : a list of array, where each array contains
        # the state information at each time of the trajectory
        self.expertPolicy = expertFE
        #self.getExpertFE = expertFE
        self.IRLFeaureSize = featsize
        self.num_states = num_states
        self.num_frames = num_frames
        self.behavior = behavior
        self.epsilon = epsilon # termination when t<0.1
        self.randomT = np.linalg.norm(np.asarray(self.expertPolicy)-np.asarray(self.randomPolicy)) #norm of the diff in expert and random
        self.policiesFE = {self.randomT:self.randomPolicy} # storing the policies and their respective t values in a dictionary
        print ("Expert - Random at the Start (t) :: " , self.randomT) 
        self.currentT = self.randomT
        self.minimumT = self.randomT
        self.weightdistanceList = []
        self.nn_param = [512, 1024 , 128]
        self.params = {
        "batchSize": 100,
        "buffer": 50000,
        "nn": nn_param
        }


        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(1,1,1)

   #given the file name, reads the dictionary created by eth_util.py and upadates the expertFE in the form of a vector.
    #this method takes expert trajectory from 3rd party tracking videos.
    def getExpertFE(self):

        gamma = .9
        trajFilelist = None
        expertFE = np.zeros([44])
        counter = np.zeros([44])
        print expertFE.sizeFRAMES
        if not os.path.isdir(self.expertTrajFile):

            print ("The trajectories do not exist.")

        for ( root, dirnames, filenames ) in os.walk(self.expertTrajFile):

            trajFilelist = filenames
            break

        totaltraj = len(trajFilelist)
        print "started"
        for i in range(len(trajFilelist)): #looping through all the trajectory files
            #print i
            timestep = 0
            singleRunFE = np.zeros([44])
            filename = os.path.join(self.expertTrajFile,trajFilelist[i])
            #print filename
            with open(filename , 'r') as f:  #opening a single file of trajectory
                for line in f:              #looping through the lines of a single trajectory file
                    state_list = line.strip().split(',')
                    state_arr = np.array(state_list)
                    state_arr = state_arr.astype(np.float)
                    for i in range(state_arr.size):  #looping through the values of a single sensor reading array
                        if not math.isnan(state_arr[i]):
                            singleRunFE[i] = singleRunFE[i]+ (math.pow(gamma,timestep)*state_arr[i])
                            counter[i]+=1
                    timestep+=1
            expertFE = np.add(expertFE,singleRunFE)
        expertFE = np.divide(expertFE,totaltraj)
        print "Expert Feature Expectation generated."
        self.expertPolicy = expertFE



    def getRLAgentFE(self, W, i): #get the feature expectations of a new poliicy using RL agent

        #IRL_helper() - returns the filename in which the state dictionary of the trained model is stored
        model_dict_file = IRL_helper(W, self.behavior, self.num_frames, i , self.IRLFeaureSize) # train the agent and save the model in a file used below
        #saved_model = 'saved-models_'+self.behavior+'/evaluatedPolicies/'+str(i)+'-164-150-100-50000-'+str(self.num_frames)+'.h5' # use the saved model to get the FE
        #model = neural_net(self.num_states, [164, 150], saved_model)

        # *** no saved model as of now ***
        print "Model dict :",model_dict_file
        return  play(W,saved_model = model_dict_file , nnParams=self.nn_param , featsize = self.IRLFeaureSize)#return feature expectations by executing the learned policy

    def get_saved_model_FileName(self):

        filename = get_saved_modelFilename()
        return filename


    def policyListUpdater(self, W, i):  #add the policyFE list and differences
        tempFE = self.getRLAgentFE(W, i) # get feature expectations of a new policy respective to the input weights
        hyperDistance = np.abs(np.dot(W, np.asarray(self.expertPolicy)-np.asarray(tempFE))) #hyperdistance = t
        self.weightdistanceList.append(hyperDistance)
        self.policiesFE[hyperDistance] = tempFE
        return hyperDistance # t = (weights.tanspose)*(expert-newPolicy)
        
    def optimalWeightFinder(self):


        i = 1
        MAX_ITERATIONS = 1000
        while True:
            W = self.optimization() # optimize to find new weights in the list of policies
            print ("weights ::", W )
            f = open('weights-'+BEHAVIOR+'.txt', 'a')
            writestr =  'Iteration :' + str(i) + ' - ' + str(W)
            f.write( writestr )
            f.write('\n')
            f.close()
            print ("the distances  ::", self.policiesFE.keys())
            self.currentT = self.policyListUpdater(W, i)
            print ("Current distance (t) is:: ", self.currentT )
            with open('results/weight_dist.txt','a') as res:
                res.write(str(self.currentT)+'\n')
                res.close()

            filename = self.get_saved_model_FileName()
            path = self.behavior
            #model_path = ''
            with open('results/model_paths.txt','r') as mdl:
                model_path = mdl.read().strip()
                mdl.close()
            test_model(W,model_path,i , self.nn_param , sensor_size = self.IRLFeatureSize)
            if self.currentT <= self.epsilon or i>= MAX_ITERATIONS: # terminate if the point reached close enough
                break
            i += 1

        return W




    def optimization(self): # implement the convex optimization, posed as an SVM problem
        print 'expert policy', self.expertPolicy
        self.expertPolicy = np.squeeze(self.expertPolicy)
        self.expertPolicy = np.squeeze(self.expertPolicy)
        m = self.expertPolicy.shape[0]
        print 'MMMM',type(m)
        P = matrix(2.0*np.eye(m), tc='d') # min ||w||
        q = matrix(np.zeros(m), tc='d')
        policyList = [self.expertPolicy]
        h_list = [1]
        for i in self.policiesFE.keys():
            policyList.append(self.policiesFE[i])
            h_list.append(1)
        print 'policyList',policyList[0].shape
        print 'policyList 2', policyList[1].shape



        policyMat = np.array(policyList)
        policyMat = policyMat.astype(np.double)
        print policyMat.dtype
        #policyMat = policyMat.astype(np.double)
        policyMat[0] = -1*policyMat[0]
        G = matrix(policyMat,tc= 'd')
        h = matrix(-np.array(h_list), tc='d')
        print "P :",P
        print "q : ",q
        print "m : ",m
        print "G : ",G
        print "h : ",h
        sol = solvers.qp(P,q,G,h)
        print sol['status']
        weights = np.squeeze(np.asarray(sol['x']))
        norm = np.linalg.norm(weights)
        weights = weights/norm
        plt.bar(range(len(weights)),weights)
        plt.show(block=False)
        return weights # return the normalized weights


#this helper method takes the FE expectation from a player playing the ball environment itsel
def getExpertFE(NO_OF_TRIALS,nn_params):


    return play_User(NO_OF_TRIALS,nn_params)

            
if __name__ == '__main__':
    logger = logging.getLogger()


    #to make changes in the featuresize change below
    #in the main of learning.py line 15

    FEATSIZE = 5
    logger.setLevel(logging.INFO)
    #randomPolicyFE = [ 7.74363107 , 4.83296402 , 6.1289194  , 0.39292849 , 2.0488831  , 0.65611318 , 6.90207523 , 2.46475348]
    nn_param = [32, 128 ,16]
    params = {
        "batchSize": 100,
        "buffer": 50000,
        "nn": nn_param
    }
    expertFE = np.asarray([np.genfromtxt('newFTFile2_1feat.txt')])
    #expertFE = np.divide(expertFE) #500 because size of the current environment is 500 x 500
    intiWeights = np.random.rand(FEATSIZE)
    randomPolicyFE = play(intiWeights , nnParams= nn_param , featsize= FEATSIZE)
    print "Generated random FE"

    print "Expert FE :",expertFE
    #trajdir = "expert_traj2"
    epsilon = .001
    print "Start agent"
    if torch.device("cuda"):
        print "Using CUDA"

    trajdir = None
    print "randompolicy", randomPolicyFE
    print "expertpolicy", expertFE
    irlearner = irlAgent(randomPolicyFE, expertFE  , epsilon, NUM_STATES, FRAMES, BEHAVIOR , FEATSIZE)
    #irlearner.getExpertFE()
    print (irlearner.optimalWeightFinder())

