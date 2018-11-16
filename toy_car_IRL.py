# IRL algorith developed for the toy car obstacle avoidance problem for testing.
import numpy as np
import logging
import torch
import scipy
#import ballenv_pygame as BE
from playing import play   #get the RL Test agent, gives out feature expectations after 2000 frames
from playing import test_model
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
FRAMES = 1000 # number of RL training frames per iteration of IRL

class irlAgent:
    def __init__(self, randomFE, expertTrajFile, epsilon, num_states, num_frames, behavior):
        self.randomPolicy = randomFE
        self.expertTrajFile = expertTrajFile #directory that contains the expert trajetories

        # The trajectories are of the of the format : a list of array, where each array contains
        # the state information at each time of the trajectory
        self.expertPolicy = None
        self.getExpertFE()

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



        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(1,1,1)

   #given the file name, reads the dictionary created by eth_util.py and upadates the expertFE in the form of a vector.

    def getExpertFE(self):

        gamma = .9
        trajFilelist = None
        expertFE = np.zeros([44])
        counter = np.zeros([44])
        print expertFE.size
        if not os.path.isdir(self.expertTrajFile):

            print ("The trajectories do not exist.")

        for ( root, dirnames, filenames ) in os.walk(self.expertTrajFile):

            trajFilelist = filenames
            break

        totaltraj = len(trajFilelist)
        print "started"
        for i in range(len(trajFilelist)):
            #print i
            timestep = 0
            singleRunFE = np.zeros([44])
            filename = os.path.join(self.expertTrajFile,trajFilelist[i])
            #print filename
            with open(filename , 'r') as f:
                for line in f:
                    state_list = line.strip().split(',')
                    state_arr = np.array(state_list)
                    state_arr = state_arr.astype(np.float)
                    for i in range(state_arr.size):
                        if not math.isnan(state_arr[i]):
                            singleRunFE[i] = singleRunFE[i]+ (math.pow(gamma,timestep)*state_arr[i])
                            counter[i]+=1
                    timestep+=1
            expertFE = np.add(expertFE,singleRunFE)
        expertFE = np.divide(expertFE,totaltraj)
        print "Expert Feature Expectation generated."
        self.expertPolicy = expertFE



    def getRLAgentFE(self, W, i): #get the feature expectations of a new poliicy using RL agent

        IRL_helper(W, self.behavior, self.num_frames, i) # train the agent and save the model in a file used below
        #saved_model = 'saved-models_'+self.behavior+'/evaluatedPolicies/'+str(i)+'-164-150-100-50000-'+str(self.num_frames)+'.h5' # use the saved model to get the FE
        #model = neural_net(self.num_states, [164, 150], saved_model)

        # *** no saved model as of now ***
        return  play(W,saved_model = None)#return feature expectations by executing the learned policy

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
        f = open('weights-'+BEHAVIOR+'.txt', 'w')
        i = 1

        while True:
            W = self.optimization() # optimize to find new weights in the list of policies
            print ("weights ::", W )
            f.write( str(W) )
            f.write('\n')
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
            test_model(W,model_path,i)
            if self.currentT <= self.epsilon: # terminate if the point reached close enough
                break
            i += 1
        f.close()
        return W




    def optimization(self): # implement the convex optimization, posed as an SVM problem
        m = len(self.expertPolicy)
        P = matrix(2.0*np.eye(m), tc='d') # min ||w||
        q = matrix(np.zeros(m), tc='d')
        policyList = [self.expertPolicy]
        h_list = [1]
        for i in self.policiesFE.keys():
            policyList.append(self.policiesFE[i])
            h_list.append(1)
        policyMat = np.matrix(policyList)
        policyMat[0] = -1*policyMat[0]
        G = matrix(policyMat, tc='d')
        h = matrix(-np.array(h_list), tc='d')
        sol = solvers.qp(P,q,G,h)

        weights = np.squeeze(np.asarray(sol['x']))
        norm = np.linalg.norm(weights)
        weights = weights/norm
        return weights # return the normalized weights
                
            
if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    #randomPolicyFE = [ 7.74363107 , 4.83296402 , 6.1289194  , 0.39292849 , 2.0488831  , 0.65611318 , 6.90207523 , 2.46475348]

    intiWeights = np.random.rand(44)
    randomPolicyFE = play(intiWeights)
    print "Generated random FE"
    # ^the random policy feature expectations
    expertPolicyYellowFE = [7.5366e+00,  4.6350e+00  , 7.4421e+00, 3.1817e-01,  8.3398e+00,  1.3710e-08,  1.3419e+00 ,  0.0000e+00]
    # ^feature expectations for the "follow Yellow obstacles" behavior
    expertPolicyRedFE = [7.9100e+00, 5.3745e-01,  5.2363e+00, 2.8652e+00,  3.3120e+00, 3.6478e-06, 3.82276074e+00  , 1.0219e-17] 
    # ^feature expectations for the follow Red obstacles behavior
    expertPolicyBrownFE = [5.2210e+00,  5.6980e+00,  7.7984e+00,  4.8440e-01, 2.0885e-04, 9.2215e+00, 2.9386e-01 , 4.8498e-17]
    # ^feature expectations for the "follow Brown obstacles" behavior
    expertPolicyBumpingFE = [  7.5313e+00, 8.2716e+00, 8.0021e+00, 2.5849e-03 ,2.4300e+01 ,9.5962e+01 ,1.5814e+01 ,1.5538e+03]
    # ^feature expectations for the "nasty bumping" behavior
    
    trajdir = "expert_traj"
    epsilon = 0.1
    print "Start agent"
    if torch.device("cuda"):
        print "Using CUDA"
    irlearner = irlAgent(randomPolicyFE, trajdir  , epsilon, NUM_STATES, FRAMES, BEHAVIOR)
    irlearner.getExpertFE()
    print (irlearner.optimalWeightFinder())

