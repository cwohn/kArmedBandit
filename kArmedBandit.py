import numpy as np
import random as rand
import math
from matplotlib import pyplot as plt


#A Bandit can give out rewards using a normal distribution based on its mean and standard deviation. 
class Bandit:

    def __init__(self, mean, std):
        self.mean = mean
        self.std=std #Standard Deviation

    #After the lever is pulled, some reward is given
    #This function returns the reward, which is a sample from the normal distribution
    def getReward(self):
        ret = np.random.normal(self.mean, self.std)
        return 0 if ret < 0 else ret

    def getMean(self):
        return self.mean

    def getStd(self):
        return self.std

#The mean of a nonstationary bandit undergoes a small, random change every time it is choosen
class Bandit_nonstationary:

    def __init__(self, mean=50, std=10):
        self.mean = mean
        self.std=std #Standard Deviation

    #After the lever is pulled, some reward is given
    #This function randomly determines that reward based on standard deviation 
    #It also updates the mean randomly
    #The return is the reward
    def getReward(self):
        ret = np.random.normal(self.mean, self.std)
        self.mean += rand.choice([-1, 1]) if rand.random() < .3 else 0 
        return 0 if ret < 0 else ret

    def getMean(self):
        return self.mean

    def getStd(self):
        return self.std


#The agent decides which lever to pull
#This class is a basic version of the agent
class Agent:

    bandits = [] #list of bandits to choose from
    values = [] #the estimated 'value' (expected reward) of the bandit at this index
    timesChosen = [] #Number of times the badnit at this index has been selected
    reward = 0 #Total reward so far
    rewards = [] #List of rewards in the order they were recived (i.e. it is indexed by the time step)
    epsilon = .1 #Probability that the agent will decide to explore rather than exploit
    time = 0 #The current timestep
    greedy = 0 #Number of times the agent has choosen greedily so far

    def __init__(self, k):
        self.k = k
        for i in range(k): #populate bandits and rewards
            mean = rand.randint(20, 90)
            self.bandits.append(Bandit(mean, 1))
            self.values.append(0)
            self.timesChosen.append(0)

    #Begin the main loop for the weighted version of this bandit 
    #Pull the lever 100,000 times, updating the values each time 
    #Afterwards, report on the results 
    def go_weighted(self):
        while self.time < 100000:
            self.pullLever(self.chooseLever_weighted())
            self.time +=1
        print(self.greedy)
        print(self.values)
        tmp = []
        for curr in self.bandits:
            tmp.append(curr.getMean())
        print(tmp)
        plt.plot(self.rewards, '.')
        plt.show()
        plt.plot(self.timesChosen, '.')
        plt.plot(self.values, '.')
        plt.show()
        return(self.reward)

    #Begin the main loop for the epsilon greedy version of this bandit 
    #Pull the lever 100,000 times, updating the values each time 
    #Afterwards, report on the results 
    def go_eGreedy(self):
        while self.time < 1000:
            if self.time % 100 == 0:
                self.epsilon -= .01
            self.pullLever(self.chooseLever_eGreedy())
            self.time +=1
        print(self.greedy)
        print(self.values)
        tmp = []
        for curr in self.bandits:
            tmp.append(curr.getMean())
        print(tmp)
        plt.plot(self.rewards, '.')
        plt.show()
        plt.plot(self.timesChosen, '.')
        plt.plot(self.values, '.')
        plt.show()
        return(self.reward)

    #Select a bandit using the epsilon-greedy method:
    #Choose the lever that has the highest expect reward with probability 1-epsilon
    #Choose randomly with probability epsilon
    def chooseLever_eGreedy(self):
        if rand.random() <= self.epsilon:
            return rand.randint(0, len(self.bandits)-1)#random bandit
        self.greedy+=1
        return self.values.index(max(self.values))

    #Select a bandit using the weighted method:
    #Probabilty of selecting a given bandit is proportional to its value divided by the total value
    def chooseLever_weighted(self):
            valueSum = len(self.values) #Not 0 in order to prevent it from summing to zero. If all elements are zero, then there is an equal non-zero chance of selecting any element
            weightedProb = []

            #Compute the sum of the value of every bandit
            for curr in self.values: 
                valueSum+=curr
          
            #Compute the weighted probability of selecting every bandit 
            for curr in self.values:
                weightedProb.append((curr+1)/valueSum)
            return int(np.random.choice(range(len(self.bandits)), 1, p=weightedProb))


#This nonstationary agent is designed to work with non-stationary Bandit problems
#This agnt always uses epsilon-greedy method of choosing a lever, but it can update the "value" of each lever in two different ways:
#1) Average -- this is the "normal" approach, as in a stationary bandit problem. The "value" is the average reward so far
#2) Weighted -- Shift the the value "in the direction" of the current reward, weighted by alpha, the step-size parameter 
class Agent_nonstationary:
    bandits = [] #list of bandits to choose from
    values = [] #the estimated 'value' (expected reward) of the bandit at this index
    timesChosen = [] #Number of times this the index has been selected
    reward = 0 #Total reward so far
    rewards = [] #List of rewards in the order they were recived (i.e. it is indexed by the time step)
    epsilon = .1 #Probability that the agent will decide to explore rather than exploit
    alpha = .1 #Step Size parameter (used in value updating) 
    time = 0 #Current timestep
    

    def __init__(self, k):
        self.reset()
        self.k = k
        for i in range(k): #populate bandits and rewards
            self.bandits.append(Bandit_nonstationary())
            self.values.append(0)
            self.timesChosen.append(0)

            
    def go_avg(self):
        while self.time < 10000:
            self.pullLever_avg(self.chooseLever_eGreedy())
            self.time +=1
        plt.plot(self.rewards, '.')
        plt.show()
        return(self.reward)

    def go_weighted(self):
        while self.time < 10000:
            self.pullLever_weighted(self.chooseLever_eGreedy())
            self.time +=1
        plt.plot(self.rewards, '.')
        plt.show()
        return(self.reward)
    


    def chooseLever_eGreedy(self):
        if rand.random() <= self.epsilon:
            return rand.randint(0, len(self.bandits)-1)#random bandit
        return self.values.index(max(self.values))

    
    #Update the values by computing the (incremental) average reward
    def pullLever_avg(self, index):
        currReward = self.bandits[index].getReward()
        self.reward += currReward
        self.timesChosen[index] += 1
        self.values[index] += (1/self.timesChosen[index])*(currReward-self.values[index])
        self.rewards.append(currReward)

    #Update the values by shifting the value "in the direction" of the current reward. 
    #The size of this shift is weighted based alpha, the step size parameter
    def pullLever_weighted(self, index):
        currReward = self.bandits[index].getReward()
        self.reward += currReward
        self.timesChosen[index] += 1
        self.values[index] += self.alpha*(currReward-self.values[index])
        self.rewards.append(currReward)

    #Reset all the variables to their default 
    #[2019 me: I vaugly remember that I was encounter problems with agents retaing information through multiple episodes. I used this function to fix that.]
    #[I think I could fix the problem now in a better way (I was missuing 'global' variables which don't work in Python the way they do in Java)]
    def reset(self):
        plt.gcf().clear()
        self.bandits = []
        self.values = []
        self.timesChosen = [] 
        self.reward = 0
        self.rewards = []
        self.epsilon = .1 
        self.alpha = .1  
        self.time = 0


#This is the "best"/most "advanced" type of Agent in this project
class Agent_gradient:

    def __init__(self, k=10):
        self.k=k
        self.bandits = []
        self.prefrences = [] #Preference for a given actions
        self.timesChosen = [] #Number of times a given action has been chosen
        self.reward = 0 #Total reward sp far
        self.rewards = [] #Rewards at each timestep
        self.alpha = .1  #step-size constant
        self.time = 0
        for i in range(k): #populate lists
            mean = rand.randint(20, 90)
            self.bandits.append(Bandit(mean, 1))
            self.prefrences.append(0)
            self.timesChosen.append(0)

    #Return the probability of selecting lever at index a. 
    #maybe use list instead of function
    def Pr(self, a): 
        numerator = math.exp(self.prefrences[a])
        denominator = 0
        for i in range(len(self.bandits)):
            denominator += math.exp(self.prefrences[i])
        return numerator/denominator


    def chooseLever(self):
       probs = []
       for i in range(len(self.bandits)):
           probs.append(self.Pr(i))
       return int(np.random.choice(range(len(self.bandits)), 1, p=probs))

    def pullLever(self, index):
        currReward = self.bandits[index].getReward()
        self.reward += currReward
        self.timesChosen[index] += 1
        self.rewards.append(currReward)
        self.prefrences[index] += self.alpha*(currReward-(self.reward/self.time))*(1-self.Pr(index))
        for i in range(len(self.bandits)):
            if i == index:
                continue
            self.prefrences[i] -= self.alpha*(currReward-(self.reward/self.time))*(self.Pr(i))
                                                                              

    def go(self, n=100):
        while self.time <= n:
            self.time+=1
            self.pullLever(self.chooseLever())
            
        plt.plot(self.rewards)
        plt.show()
