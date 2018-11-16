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


        self.fc1 = nn.Linear(num_sensors,params[0])
        self.drop1 = nn.Dropout(.2)
        self.fc2 = nn.Linear(params[0],params[1])
        self.drop2 = nn.Dropout(.2)
        self.fc3 = nn.Linear(params[1],num_actions)

    def forward(self,x):

        x = Variable(torch.from_numpy(x))
        x = x.type(torch.cuda.FloatTensor)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)

        return x

'''
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        
        

def neural_net(num_sensors, params, load=''):
    model = Sequential()

    # First layer.
    model.add(Dense(
        params[0], init='lecun_uniform', input_shape=(num_sensors,)
    ))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Second layer.
    model.add(Dense(params[1], init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Output layer.
    model.add(Dense(3, init='lecun_uniform')) #!
    model.add(Activation('linear'))

    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)

    if load:
        model.load_weights(load)

    return model


def lstm_net(num_sensors, load=False):
    model = Sequential()
    model.add(LSTM(
        output_dim=512, input_dim=num_sensors, return_sequences=True
    ))
    model.add(Dropout(0.2))
    model.add(LSTM(output_dim=512, input_dim=512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=3, input_dim=512)) #!
    model.add(Activation("linear"))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")

    return model
'''
