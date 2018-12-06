"""
The design of this comes from here:
http://outlace.com/Reinforcement-Learning-Part-3/
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class Neural_net(nn.Module):

    def __init__(self,num_sensors,params,num_actions):

        super(Neural_net,self).__init__()

        #print 'Number of sensors',num_sensors
        self.fc1 = nn.Linear(num_sensors,params[0])
        self.bn1 = nn.BatchNorm1d(params[0])
        self.fc2 = nn.Linear(params[0],params[1])
        self.bn2 = nn.BatchNorm1d(params[1])
        #self.fc3 = nn.Linear(params[1],params[2])
        self.bn3 = nn.BatchNorm1d(params[1])
        self.fc3 = nn.Linear(params[1], num_actions)





    def forward(self,x):

        #x = np.transpose(x)

        x = Variable(torch.from_numpy(x))
        x = x.type(torch.cuda.FloatTensor)


        #print x.size()

        # if len(x.size()) == 2:
        #
        #     x = F.relu(self.bn1(self.fc1(x)))
        #     x = F.relu(self.bn2(self.fc2(x)))
        #     x = F.relu(self.bn3(self.fc3(x)))
        #     x = self.fc4(x)
        #
        #
        # else:

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.fc3(x)
        return x


'''
        self.fc1 = nn.Linear(num_sensors,params[0])
        self.drop1 = nn.Dropout(.2)
        self.fc2 = nn.Linear(params[0],params[1])
        self.drop2 = nn.Dropout(.2)
        self.fc3 = nn.Linear(params[1],params[2])
        self.drop3 = nn.Dropout(.2)
        self.fc4 = nn.Linear(params[2], num_actions)

    def forward(self,x):

        x = Variable(torch.from_numpy(x))
        x = x.type(torch.cuda.FloatTensor)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = F.relu(self.fc3(x))
        x = self.drop3(x)
        x = self.fc4(x)


        return x

'''
