import numpy as np
import random
import csv
from nn import Neural_net as neural_net
import os.path
import timeit
import math
import matplotlib.pyplot as plt
import os
import datetime
import torch.optim as optim
import torch.nn as NN
import torch
#import ballenv_pygame as BEG
from flat_game import ballgamepyg as BE
NUM_INPUT = 5
GAMMA = 0.99  # Forgetting.
TUNING = False  # If False, just use arbitrary, pre-selected params.
TRAIN_FRAMES = 1000 # to train for 100K frames in total
UPDATE_TARGET = 20 #update the target network after 20 frames
import playing




#In my code the model is not an external module rather a part of the game environment itself, called the agentBrain

#Either you create a model from scratch and train it. Or load a model. If you have a pretrained model send in the filepath
#and the model should be read into the ballgamepyg.createboardIRL.agentBrain on initialization.

#might be subject to change, if things become to difficult

#model = path to model or none if there is no model

def train_net(model_path, params, weights, path, trainFrames,i,FEATSIZE, irl = True):


    print "start Training . . ."
    filename = params_to_filename(params)
    curDay = str(datetime.datetime.now().date())
    curtime = str(datetime.datetime.now().time())
    basePath = 'saved-models_'+ path +'/evaluatedPoliciesTest/'
    subPath = curDay + '/' + curtime + '/'
    curDir = basePath + subPath
    os.makedirs(curDir)
    if os.path.exists(curDir):
        print "YES"
    observe = 100  # Number of frames to observe before training.
    epsilon = .5
    train_frames = trainFrames  # Number of frames to play. 
    batchSize = params['batchSize']
    buffer = params['buffer']

    # Just stuff used below.
    max_car_distance = 0
    car_distance = 0
    t = 0
    data_collect = []
    replay = []  # stores tuples of (S, A, R, S').

    loss_log = []
    loss_plot = []


    #Make changes here - read from a file
    # Create a new game instance.


    game = BE.createBoardIRL(sensor_size=FEATSIZE,display= False, saved_model=model_path , weights = weights , hidden_layers= params['nn'])


    #create the target network
    targetNetwork = neural_net(FEATSIZE , params['nn'],4)
    stepcounter = 0 #keeps track of how many steps the learning network has taken ahead of the target network
    UPDATETARGET = 50 #no of times after which the target network needs to be updated
    targetNetwork.load_state_dict(game.agentBrain.state_dict())
    targetNetwork.eval()
    targetNetwork.cuda()

    criterion = NN.SmoothL1Loss() #huber loss
    #criterion = NN.MSELoss()
    optimizer = optim.RMSprop(game.agentBrain.parameters() , lr = .001)
    # Get initial state by doing nothing and getting the state.
    #_, state, temp1 = game_state.frame_step((2))
    game.reset()

    #after this point state is referred to as the sensor_readings
    state = game.sensor_readings
    # Let's time it.
    start_time = timeit.default_timer()

    # Run the frames.
    while t < train_frames:
        if t%2000==0:
            print t
            plt.figure()
            plt.plot(loss_plot)
            pltfile = 'saved_plots/'+'smallexperiment/'+'featsize-'+str(FEATSIZE)+str(i)+'-'+'epoch'+str(t)
            plt.savefig(pltfile+'.png')
            #playing.test_model(weights, )
        t += 1
        car_distance += 1

        # Choose an action. so, as long as t < observe we take random actions?
        if random.random() < epsilon or t < observe:
            actionIndex = np.random.randint(0, 3)  # random #3
            action = game.agent_action_to_WorldAction(actionIndex)
        else:
            # Get Q values for each action.

            actionIndex = game.gen_action_from_agent()

            # *** agent_action_to_WorldAction() **** converts the action(which is basically an index that points to the action with the best qvalue according to the neural net)

            # *** to an action that is actually used in the game environment. An (x,y) tuple, which depicts the movement of the agent in the game environment

             #  action [2]   ---- > action [(3,4)]
            action = game.agent_action_to_WorldAction(actionIndex)
            #qval = model.predict(state, batch_size=1)
            #action = (np.argmax(qval))  # this step is already done in the method : gen_action_from_agent()
            #print ("action under learner ", action)

        # Take action, observe new state and get our treat.
        new_state,reward, done, _ = game.step(action)
        new_state = game.sensor_readings
        # Experience replay storage.
        replay.append((state, actionIndex, reward, new_state))

        # If we're done observing, start training.
        if t > observe:

            # If we've stored enough in our buffer, pop the oldest.
            if len(replay) > buffer:
                replay.pop(0)

            # Randomly sample our experience replay memory
            #print len(replay) , batchSize
            minibatch = random.sample(replay, batchSize)

            # Get training values.
            X_train, y_train = process_minibatch(minibatch, game.agentBrain,targetNetwork) #instead of the model

            #print 'Xtrain',X_train
            #print 'y_train', y_train

            #print "Printing from train and test in learning.py :"
            #print type(X_train) , X_train.size
            #print type(y_train) , y_train.size
            # Train the model on this batch.
            #history = LossHistory()


            y_train = torch.from_numpy(y_train)

            y_train = y_train.type(torch.cuda.FloatTensor)
            #chagnes to be done from here
            #change the train method from keras to pytorch





            #X_train has to be a tensor of size n x 44 x 1
            #y_train has to be a tensor of size n x 1 x 1 ??


            output = game.agentBrain(X_train)
            loss = criterion(output , y_train)

            optimizer.zero_grad()
            loss.backward()
            #print loss.item()
            #print type(loss.item())
            optimizer.step()
            stepcounter+=1

            if stepcounter==UPDATETARGET:
                targetNetwork.load_state_dict(game.agentBrain.state_dict())
                stepcounter = 0
                #print 'Updated'


            loss_log.append([t,loss.item()])
            loss_plot.append(loss.item())

        # Update the starting state with S'.
        state = new_state

        # Decrement epsilon over time.
        if epsilon > 0.1 and t > observe:
            epsilon -= (1/train_frames)

        # We died, so update stuff.
        if done == True:
            # Log the car's distance at this T.
            data_collect.append([t, car_distance , ])

            # Update max.
            if car_distance > max_car_distance:
                max_car_distance = car_distance

            # Time it.
            tot_time = timeit.default_timer() - start_time
            fps = car_distance / tot_time

            # Output some stuff so we can watch.
            #print("Max: %d at %d\tepsilon %f\t(%d)\t%f fps" %
                  #(max_car_distance, t, epsilon, car_distance, fps))

            # Reset.
            game.reset()
            car_distance = 0
            start_time = timeit.default_timer()

        # Save the model 
        if t % 2000 == 0:

            #game.agentBrain.save_weights('saved-models_'+ path +'/evaluatedPolicies/'+str(i)+'-'+ filename + '-' +
             #                  str(t) + '.h5',
             #                  overwrite=True)
            torch.save(game.agentBrain.state_dict(),'saved-models_'+ path +'/evaluatedPoliciesTest/'+subPath+str(i)+'-'+ filename + '-' + str(t) + '.h5', )
            savedFilename = 'saved-models_'+ path +'/evaluatedPoliciesTest/'+subPath+str(i)+'-'+ filename + '-' + str(t) + '.h5'
            with open('results/model_paths.txt','w') as ff:
                ff.write('saved-models_'+ path +'/evaluatedPoliciesTest/'+subPath+str(i)+'-'+ filename + '-' + str(t) + '.h5\n')
                ff.close()
            print("Saving model %s - %d" % (filename, t))

    # Log results after we're done all frames.
    log_results(filename, data_collect, loss_log,i)
    #print "Testing the model :"

    return savedFilename




def log_results(filename, data_collect, loss_log,i):
    # Save the results to a file so we can graph it later.
    with open('results/sonar-frames/learn_data-' + filename +'_'+str(i)+ '.csv', 'w') as data_dump:
        wr = csv.writer(data_dump)
        wr.writerows(data_collect)

    with open('results/sonar-frames/loss_data-' + filename +'_'+str(i)+ '.csv', 'w') as lf:
        wr = csv.writer(lf)
        #print loss_log
        #print type(loss_log)
        for loss_item in loss_log:
            wr.writerow(loss_item)


def process_minibatch(minibatch, model ,targetNetwork):
    #here the model is the game.agentBrain
    """This does the heavy lifting, aka, the training. It's super jacked."""
    X_train = []
    y_train = []
    #print "Starting process_minibatch method. . ."
    # Loop through our batch and create arrays for X and y
    # so that we can fit our model at every step.
    for memory in minibatch:
        # Get stored values.
        old_state_m, action_m, reward_m, new_state_m = memory
        # Get prediction on old state.


        old_qval = model(old_state_m)
        # Get prediction on new state.
        newQ = targetNetwork(new_state_m)
        # Get our best move. I think?
        #print "The newQ :", newQ


        newQ = newQ.cpu().detach().numpy()
        #print type(newQ)
        maxQ = np.max(newQ)
        y = np.zeros((1, 4)) #3
        old_qval = old_qval.cpu().detach().numpy()
        y[:] = old_qval[:]
        # Check for terminal state.
        #if reward_m != -500:  # non-terminal state
            #update = (reward_m + (GAMMA * maxQ))
        #else:  # terminal state
            #update = reward_m
        if new_state_m[3] == 1:
            update = reward_m
        else:  # non-terminal state
            update = (reward_m + (GAMMA * maxQ))
        # Update the value for the action we took.
        y[0][action_m] = update
        X_train.append(old_state_m.reshape(NUM_INPUT,))
        y_train.append(y.reshape(4,)) #3

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train


def params_to_filename(params):
    return str(params['nn'][0]) + '-' + str(params['nn'][1])+ '-'+ str(params['nn'][1]) + '-' + \
            str(params['batchSize']) + '-' + str(params['buffer'])


def launch_learn(params):
    filename = params_to_filename(params)
    print("Trying %s" % filename)
    # Make sure we haven't run this one.
    if not os.path.isfile('results/sonar-frames/loss_data-' + filename + '.csv'):
        # Create file so we don't double test when we run multiple
        # instances of the script at the same time.
        open('results/sonar-frames/loss_data-' + filename + '.csv', 'a').close()
        print("Starting test.")
        # Train.
        model = neural_net(NUM_INPUT, params['nn'])
        train_net(model, params)
    else:
        print("Already tested.")


def get_saved_modelFilename():
    nn_param = [512, 1024 , 128]
    params = {
        "batchSize": 100,
        "buffer": 50000,
        "nn": nn_param
    }
    return params_to_filename(params)



def IRL_helper(weights, path, trainFrames, i,FEATSIZE):
    nn_param = [512, 1024 , 128]
    params = {
        "batchSize": 100,
        "buffer": 50000,
        "nn": nn_param
    }
    ####model = neural_net(NUM_INPUT, nn_param , num_actions=4)

    #train_net() - the first parameter used to be the model, but now as because the model
    # is being created inside the game environment, pass in either the path to a '/home/abhisek/Study/Robotics/toySocialNav/saved-models_red/eval'/home/abhisek/Study/Robotics/toySocialNav/saved-models_red/eval'/home/abhisek/Study/Robotics/toySocialNav/saved-models_red/eval'/home/abhisek/Study/Robotics/toySocialNav/saved-models_red/evaluatedPolicies/17-512-1024-100-50000-10011.h5'uatedPolicies/17-512-1024-100-50000-10011.h5'uatedPolicies/17-512-1024-100-50000-10011.h5'uatedPolicies/17-512-1024-100-50000-10011.h5'saved model or None
    #train_net method returns the path in which the model dictionary is saved
    return train_net(None, params, weights, path, trainFrames, i,FEATSIZE)

'/home/abhisek/Study/Robotics/toySocialNav/saved-models_red/evaluatedPolicies/17-512-1024-100-50000-10011.h5'


if __name__ == "__main__":
    weights = [-.4, 0, 0 , 0 ,0]
    path = 'train_from_main'
    FEATSIZE = 4
    if TUNING:
        param_list = []
        nn_param = [32 , 32]
        params = {
        "batchSize": 100,
        "buffer": 50000,
        "nn": nn_param
    }
    else:
        nn_param = [32,32]
        params = {
        "batchSize": 100,
        "buffer": 50000,
        "nn": nn_param
        }

        #model_path = '/home/abhisek/Study/Robotics/toySocialNav/2-512--1024-128-100-50000-98000.h5'

        train_net(None, params, weights, path, 1000000,111,FEATSIZE)
