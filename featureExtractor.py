
#given a state information from the ballgame environment, this file is dedicated to extract features from it

import numpy as np
import math
import pygame

#from flat_game import ballgamepyg as BE
'''
the features extracted here are suggested by the paper : 
    Inverse Reinforcement Learning ALgorithms and features for robot navigation in Crowds

    1.Density features : The number of human agents n contained in a circle of radius r centered around the robot Phi(d)
    2.Speed+orientations : relative speed and orientation of the pedestrians. Phi(s), Phi(o)
    3.Velocity Features : AVERAGE magnitude of obstacles moving towards the robot.
                        Classify each obstacle into an orientation bin
                        Calculate the avg speed of the obstacles in each of the bin
    4.Default Cost features : always set to 1
    5.Social force features : Phi(s)
    6.Social + Relative Velocity forces : 

State information description:
    state : list
    state[0] - tuple containing agent current position
    state[1] - tuple containing goal position.
    state[2] - distance from goal
    state[3] - done?
    state[4 - end] - tuple obstacle position
    
'''


ENVIRONMENT_SIZE = 500
#returns the distance between the obstacle and the agent subtracting the radius of both
def calcDistance(obstacle , agent_pos , agent_rad):

    dist = np.linalg.norm((obstacle.x-agent_pos[0],obstacle.y-agent_pos[1]))

    return dist-agent_rad-obstacle.rad


def unit_vector(vector):
    #print "vector"
    if np.linalg.norm(vector)>0:
        return vector / np.linalg.norm(vector)
    else:
        return vector


#calculates the angle between 2 vectors
def angle_between(v1, v2):
    #Returns the angle in radians between vectors 'v1' and 'v2'::
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))



#angle between relative_pos and relative_vel
def calcOrientationAndVelocity(obstacle , agent_pos, agent_vel):

    v1 = np.asarray([obstacle.x - agent_pos[0], obstacle.y - agent_pos[1]])
    v2 = np.asarray([obstacle.vel_x - agent_vel[0] , obstacle.vel_y - agent_vel[1]])

    relvel = np.linalg.norm(v2)
    speedBin = -1
    if relvel < 0.015:
        speedBin = 0
    elif relvel < 0.025:
        speedBin = 1
    else:
        speedBin = 2

    angle = angle_between(v1,v2)

    if angle < math.pi/4:
        return 0 , speedBin # bin1

    if angle >  math.pi/4 and angle < math.pi*3/4:
        return 1 , speedBin # bin2

    else:
        return 2 , speedBin # bin3

    return 0


#works
def densityFeatures(stateInfo, obstacle_List ,agent_rad , agent_vel):

    agent_pos = stateInfo[0]
    obstacleInfo = stateInfo[4:]
    phi_D = np.zeros(3)

    threshold1 =  101
    threshold2 =  230
    threshold3 = 1000

    for obstacle in obstacle_List:
        dist = calcDistance(obstacle, agent_pos ,agent_rad)
        #print 'Obstacle pos :',obstacle.x,obstacle.y
        #print 'Corresponding distance :', dist
        if dist < threshold3: #consider in density features
            phi_D[2]+=1
        if dist < threshold2:
            phi_D[1]+=1
        if dist < threshold1:
            phi_D[0]+=1

    return  phi_D

#
def  speedOrientationFeatures(stateInfo , obstacle_List , agent_rad , agent_vel):

    agent_pos = stateInfo[0]
    phi_O = np.zeros([3,3])

    for obstacle in obstacle_List:

        i,j = calcOrientationAndVelocity(obstacle, agent_pos , agent_vel)
        phi_O[i,j]+=1

    #rows - orientation bins
    # orientation bin 0 - going away
    # bin 1 = perpendicular
    # bin 2 = going towards
    #columns - relative speed bins
    return phi_O

def calcDistanceFromGoal(stateInfo , binDist):

    agent_pos = stateInfo[0]
    goal_pos = stateInfo[1]

    dist = np.hypot(agent_pos[0]-goal_pos[0],agent_pos[1]-goal_pos[1])

    dist = math.floor(dist/binDist)

    if dist>5:
        return np.asarray([5])
    else:
        return np.asarray([dist])

def relativeGoalPos(state):

    pos_vector = np.zeros(4)
    agent_pos= state[0]
    goal_pos = state[1]
    xindicator = goal_pos[0] - agent_pos[0]
    yindicator = goal_pos[1] - agent_pos[1]
    base_vector = [0,1]
    angle = angle_between(base_vector,(xindicator,yindicator))

    if angle < math.pi/4:
        pos_vector[0] =1
    elif angle > math.pi/4 and angle < math.pi*3/4:
        if xindicator>0:
            pos_vector[1] = 1
        else:
            pos_vector[3] = 1
    else:
        pos_vector[2] = 1

    return pos_vector



def socialForcesFeatures(stateInfo, obstacle_List , agent_rad , agent_vel):
    a = 1
    b = 10
    lam = 2
    phi_SF = np.zeros(3)
    agent_pos = stateInfo[0]
    threshold = 1
    for obstacle in obstacle_List:

        orientationBin,_ = calcOrientationAndVelocity(obstacle , agent_pos, agent_vel)
        v1 = np.asarray([obstacle.x - agent_pos[0], obstacle.y - agent_pos[1]])
        v2 = np.asarray([obstacle.vel_x - agent_vel[0] , obstacle.vel_y - agent_vel[1]])

        psi_ij = angle_between(v1,v2)
        #print 'dist',-calcDistance(obstacle, agent_pos, agent_rad)
        force_exp = a*np.exp(-calcDistance(obstacle, agent_pos, agent_rad)/b)
        Nij = calcDistance(obstacle,agent_pos, agent_rad)
        thrPart= (lam + 0.5*(1-lam)*(1+np.cos(psi_ij)))
        f_soc_ij = a*force_exp*Nij*thrPart
        #print f_soc_ij
        if f_soc_ij > threshold: #update only when force exceeds predetermined threshold
            phi_SF[orientationBin]+=f_soc_ij

    return phi_SF

def drawOrientationLines(gameDisplay ,color , agent_pos,agent_vel):

    coord = np.asarray(agent_vel)
    theta1 = math.pi/4
    roTmat = np.asarray([[math.cos(theta1) , -math.sin(theta1)],[ math.sin(theta1), math.cos(theta1)]])
    rotCoord1 = np.matmul(roTmat,coord)
    mag = 30
    pygame.draw.line(gameDisplay, color , [agent_pos[0]-mag*rotCoord1[0],agent_pos[1]-mag*rotCoord1[1]], [(agent_pos[0]+mag*rotCoord1[0]),(agent_pos[1]+mag*rotCoord1[1])] ,1)

    theta2 = -math.pi/4
    roTmat2 = np.asarray([[math.cos(theta2) , -math.sin(theta2)],[ math.sin(theta2), math.cos(theta2)]])
    rotCoord2 = np.matmul( roTmat2 , coord)
    pygame.draw.line(gameDisplay, color , [agent_pos[0]-mag*rotCoord2[0],agent_pos[1]-mag*rotCoord2[1]], [(agent_pos[0]+mag*rotCoord2[0]),(agent_pos[1]+mag*rotCoord2[1])] ,1)

def renderObstacle(gameDisplay , color , obs):

    pygame.draw.circle(gameDisplay, color, (obs.x, obs.y), obs.rad)
    mag = 5
    pygame.draw.line(gameDisplay,color, [obs.x,obs.y],[obs.x+mag*obs.vel_x , obs.y+mag*obs.vel_y],1)



def visualizeEnvironment(agent_pos,goal_pos,obstacle_List , agent_rad ,agent_vel):
    pygame.init()
    clock = pygame.time.Clock()
    height = ENVIRONMENT_SIZE
    width = ENVIRONMENT_SIZE
    red = (255,0,0)
    green = (0,255,0)
    blue = (0,0,255)
    white = (255,255,255)
    black = (0,0,0)

    gameDisplay = pygame.display.set_mode((width,height))
    gameDisplay.fill(white)
    for obs in obstacle_List:
        renderObstacle(gameDisplay , red , obs)

    # draw agent
    pygame.draw.circle(gameDisplay, black, (int(agent_pos[0]), int(agent_pos[1])), agent_rad)
    mag = 4 # increase mag if the orientation indicator is too small

    pygame.draw.circle(gameDisplay, green , (int(goal_pos[0]) , int(goal_pos[1])), 10)
    # draw agent orientation
    pygame.draw.line(gameDisplay, blue , [agent_pos[0],agent_pos[1]], [(agent_pos[0]+mag*agent_vel[0]),(agent_pos[1]+mag*agent_vel[1])] ,1)
    #draw orientation lines ??
    drawOrientationLines(gameDisplay, black , agent_pos,agent_vel)

    pygame.display.update()


def featureExtractor(state, ObstacleList,agent_vel,agent_rad):

    distance_bin = 20 # 20,40,60,80 anything more than that
    distanceFromGoal = calcDistanceFromGoal(state,distance_bin)
    goal_pos = relativeGoalPos(state)
    #dFeatures = densityFeatures(state, ObstacleList , agent_rad, 10)
    #orSpeedFeat = speedOrientationFeatures(state, ObstacleList , agent_rad , agent_vel)
    #sfFeatures = socialForcesFeatures(state,ObstacleList,agent_rad,agent_vel)

    #spdor = np.reshape(orSpeedFeat,orSpeedFeat.shape[0]*orSpeedFeat.shape[1])
    finalFeat = np.concatenate((distanceFromGoal,goal_pos))#,spdor,sfFeatures))
    #finalFeat = distanceFromGoal
    return finalFeat
    #print finalFeat.shape

if __name__=='__main__':

    #test the features
    obstacle_Rad = 10
    obstacle_loc_list = [(11,30),(200,150),(180,150),(1,90)]
    obstacle_vel_list = [(8,3),(0,3),(7,0),(1,9)]
    agent_pos = (200,200)
    goal_pos = (200,100)
    agent_vel = (0,4)
    agent_rad = 5
    obstacle_List = []
    for i in range(len(obstacle_loc_list)):
        temp_obs = BE.Obstacle(i,xpos=obstacle_loc_list[i][0],ypos=obstacle_loc_list[i][1], xvel= obstacle_vel_list[i][0] , yvel = obstacle_vel_list[i][1], radius = obstacle_Rad)
        obstacle_List.append(temp_obs)

    state = []
    state.append(agent_pos) #all the other stuff are taken from the obstacleList info
    state.append(goal_pos)
 #   print 'Densitysdd',densityFeatures(state, obstacle_List ,agent_rad , 10)
 #   print 'SpeedOrientation',speedOrientationFeatures(state, obstacle_List , agent_rad , agent_vel)
 #   print 'Socialforces',socialForcesFeatures(state,obstacle_List,agent_rad,agent_vel)
#    print 'Distance from goal',state[2]
    print featureExtractor(state,obstacle_List,agent_vel,agent_rad)
    while True:
        visualizeEnvironment(agent_pos,goal_pos ,obstacle_List,agent_rad,agent_vel)

    print featureExtractor(state,obstacle_List,agent_vel, agent_rad)







