import ballenv_pygame as BE
#from flat_game import ballgamepyg as BE
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

env = BE.createBoard(display= True, static_obstacles= 3 , static_obstacle_radius= 20)
AGENT_RAD = 10


gamma = .99
log_interval = 1000
render = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.ion()


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

def block_to_arrpos(window_size,x,y):

    a = (window_size**2-1)/2
    b = window_size
    pos = a+(b*y)+x
    return int(pos)


def get_state_BallEnv(state):

#state is a list of info where 1st position holds the position of the
#agent, 2nd the position of the goal , 3rd the distance after that,
#the positions of the obstacles in the world
    #print(state)
    window_size = 5
    block_width = 2

    window_rows = window_size
    row_start =  (window_rows-1)/2
    window_cols = window_size
    col_start = (window_cols-1)/2

    ref_state = np.zeros(4+window_size**2)
    #print(ref_state.shape)
    a = (window_size**2-1)/2
    ref_state[a+4] = 1
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

        #as of now this just measures the distance from the center of the obstacle
        #this distance has to be measured from the circumferance of the obstacle

        #new method, simulate overlap for each of the neighbouring places
        #for each of the obstacles
        obs_pos = state[i][0:2]
        obs_rad = state[i][2]
        for r in range(-row_start,row_start+1,1):
            for c in range(-col_start,col_start+1,1):
                #c = x and r = y
                temp_pos = (agent_pos[0] + c*block_width , agent_pos[1] + r*block_width)
                if checkOverlap(temp_pos,AGENT_RAD, obs_pos, obs_rad):
                    pos = block_to_arrpos(window_size,r,c)

                    ref_state[pos]=1

    #state is as follows:
        #first - tuple agent position
        #second -
    state = torch.from_numpy(ref_state).to(device)
    state = state.type(torch.cuda.FloatTensor)
    state = state.unsqueeze(0)

    return state


#returns true if there is an overlap
def checkOverlap(obj1Pos,obj1rad, obj2Pos, obj2rad):

    xdiff = obj1Pos[0]-obj2Pos[0]
    ydiff = obj1Pos[1]-obj2Pos[1]

    if (np.hypot(xdiff,ydiff)-obj1rad-obj2rad) > 0:

        return False
    else:
        return True

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
    def __init__(self, inputSize = 5 , outputSize = 9 , hidden = 128):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(inputSize, hidden)
        #self.affine1 = nn.Linear(inputSize,512)
        #self.affine2 = nn.Linear(512, 128)
        self.action_head = nn.Linear(hidden, outputSize)
        self.value_head = nn.Linear(hidden, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = F.relu(self.affine2(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values


policy = Policy(inputSize=20)
policy.cuda()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state,policy):
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

def testmodel(modelpath, iterations):

    policy = Policy(inputSize=29)
    policy.load_state_dict(torch.load(modelpath))

    policy.eval()
    policy.cuda()

    for i in policy.parameters():

        print i.shape
    env = BE.createBoard(display=  True , static_obstacles= 40)
    reward_list = []
    run_list = []

    fig = plt.figure(1)
    plt.clf()

    for i in range(iterations):

        state = env.reset()
        state = get_state_BallEnv(state)
        #state = env.sensor_readings
        done = False
        t = 0
        while not done and t < 1000:
            action = select_action(state,policy)
            t+=1
            if action != None:

                action = move_list[action]

                state, reward, done, _ = env.step(action)
                #print 'dist :',state[2]
                state = get_state_BallEnv(state)
                #state = env.sensor_readings
                env.render()

        run_list.append(i)
        reward_list.append(env.total_reward_accumulated)
        plt.plot(run_list, reward_list,color='black')
        plt.draw()
        plt.pause(.0001)
        fig.show()


#move_list = [(5,5) , (5,-5) , (5, 0) , (0,5) , (0,-5),(0,0) , (-5,5),(-5,0),(-5,-5)]
move_list = [(1,1) , (1,-1) , (1, 0) , (0,1) , (0,-1),(0,0) , (-1,1),(-1,0),(-1,-1)]
plot_interval = 10
log_interval = 10


def main():

    #****information to store the model
    filename = 'actorCriticFeaturesFull'
    curDay = str(datetime.datetime.now().date())
    curtime = str(datetime.datetime.now().time())
    basePath = 'saved-models_trainBlock' +'/evaluatedPoliciesTest/'
    subPath = curDay + '/' + curtime + '/'
    curDir = basePath + subPath
    os.makedirs(curDir)
    if os.path.exists(curDir):
        print "YES"

    #******************************


    state = env.reset()
    rewardList = []
    runList = []
    timeList = []
    fig = plt.figure(1)
    plt.clf()
    print eps
    for i_episode in range(1000000):
        running_reward = eps
        state = env.reset()
        #env.render()
        print 'Starting episode :', i_episode
        #state = get_state_BallEnv(state)
        state = env.sensor_readings
        for t in range(500):  # Don't infinite loop while learning
            action = select_action(state,policy)
            #print action
            if action!=None:
                action = move_list[action]
                #action = agent_action_to_WorldActionSimplified(action)
                #print action
                state, reward, done, _ = env.step(action)
                #state = get_state_BallEnv(state)
                state = env.sensor_readings
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
        timeList.append(float(t)/500)
        plt.plot(runList, rewardList,color='black')
        plt.plot(runList , timeList , color= 'red')
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
    #model = '/home/abhisek/Study/Robotics/toySocialNav/saved-models_trainBlock/evaluatedPoliciesTest/2018-12-16/11:40:41.466716/610-actorCritic-610.h5'
    #testmodel(model,50)
