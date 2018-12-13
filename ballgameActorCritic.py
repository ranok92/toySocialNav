import ballenv_pygame as BE
import numpy as np
from itertools import count
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import datetime
from matplotlib import pyplot as plt
import os

env = BE.createBoard(display= True,static_obstacles=15)
gamma = .99
log_interval = 1000
render = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.ion()


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

def block_to_arrpos(x,y):

    pos = 12+(5*y)+x
    return int(pos)


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

def agent_action_to_WorldActionSimplified(action):
    if action==0: #move front
        return np.asarray([0,-5])
    if action==1: #move right
        return np.asarray([5,0])
    if action==2: #move down
        return np.asarray([0,5])
    if action==3: #move left
        return np.asarray([-5,0])




class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(29, 512)
        self.affine2 = nn.Linear(512, 128)
        self.action_head = nn.Linear(128, 4)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values


policy = Policy()
policy.cuda()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    #state = torch.from_numpy(state).float().unsqueeze(0)
    '''
    for x in policy.parameters():
        #print 'One'
        #print 'x : ', torch.norm(x.data)
        if x.grad is not None:
            print 'x grad ', torch.norm(x.grad)
    print 'The state :',state
    '''
    probs ,state_value  = policy(state)
    #print 'probs :' , probs
    m = Categorical(probs)
    action = m.sample()
    #print action

    policy.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()


def finish_episode():
    R = 0
    saved_actions = policy.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards).to(device)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.item()
        policy_losses.append(-log_prob * reward)
        #print value.shape
        #print torch.tensor([r]).to(device).shape
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r]).to(device).unsqueeze(0)))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_actions[:]


plot_interval = 10
log_interval = 10
def main():

    filename = 'actorCritic'
    curDay = str(datetime.datetime.now().date())
    curtime = str(datetime.datetime.now().time())
    basePath = 'saved-models_trainBlock' +'/evaluatedPoliciesTest/'
    subPath = curDay + '/' + curtime + '/'
    curDir = basePath + subPath
    os.makedirs(curDir)
    if os.path.exists(curDir):
        print "YES"


    state = env.reset()
    rewardList = []
    runList = []

    fig = plt.figure(1)
    plt.clf()
    print eps
    for i_episode in range(1000000):
        running_reward = eps
        state = env.reset()
        #env.render()
        print 'Starting episode :', i_episode
        state = get_state_BallEnv(state)
        for t in range(1000):  # Don't infinite loop while learning
            action = select_action(state)
            #print action
            if action!=None:
                action = agent_action_to_WorldActionSimplified(action)
            #print action
                state, reward, done, _ = env.step(action)
                state = get_state_BallEnv(state)
                if i_episode%log_interval==0:
                    env.render()
                policy.rewards.append(reward)
                if done:
                    break
                running_reward += reward
            else:
                continue
            #if t%500==0:
                #print "T :",t
        #running_reward = running_reward * 0.99 + t * 0.01

        rewardList.append(env.total_reward_accumulated)
        runList.append(i_episode)
        plt.plot(runList, rewardList,color='black')
        plt.draw()
        plt.pause(.0001)
        fig.show()
        if i_episode%plot_interval==0:
            plt.savefig('saved_plots/actorCritic/plotNo{}'.format(i_episode))
        #print 'The running reward for episode {}:'.format(i_episode),running_reward
        if i_episode%log_interval==0:
            torch.save(policy.state_dict(),'saved-models_'+ 'trainBlock' +'/evaluatedPoliciesTest/'+subPath+str(i_episode)+'-'+ filename + '-' + str(i_episode) + '.h5', )

            #save the model
        finish_episode()
        #if i_episode+1 % log_interval == 0:
        #    print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
        #        i_episode, t, running_reward))
            #env.render()



if __name__ == '__main__':
    main()
