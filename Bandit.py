import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.optimize import linprog

PATH = 10
structure = []
for i in range(PATH):
	structure.append(random.randint(1,1000000))

REWARD = np.zeros(PATH)

class Bandit:
	def __init__(self, kArm=10, epsilon=0., initial=0., stepSize=0.1, trueReward=0., degree = []):
		self.k = kArm
		self.stepSize = stepSize
		self.epsilon = epsilon
		self.time = 0
		self.pathid = -1
		#The degree of destination
		self.degree = degree
		# real reward for each action
		self.qTrue = []
        # estimation for each action
		self.qEst = np.zeros(self.k)

        # # of chosen times for each action
		self.actionCount = []
		self.averageReward = 0

        # initialize real rewards with N(0,1) distribution and estimations with desired initial value
		for i in range(0, self.k):
			self.qTrue.append(np.random.randn() + trueReward)
			self.qEst[i] = initial
			self.actionCount.append(0)

		self.bestAction = np.argmax(self.qTrue)

	def getAction(self):
		# explore
		if self.epsilon > 0:
			if np.random.binomial(1, self.epsilon) == 1:
				self.pathid += 1
				if self.pathid > self.k:
					self.pathid = 0
				return self.pathid

        # exploit
		return np.argmax(self.qEst)

	def takeAction(self, action):
        # generate the reward under N(real reward, 1)
		reward = 0
		for i in range(self.k):
			if random.uniform(0,1)<=0.5 and i != action:
				reward -= self.degree[i]
		self.time += 1
		self.averageReward = (self.time - 1.0) / self.time * self.averageReward + reward / self.time
		self.actionCount[action] += 1
		# update estimation with constant step size
		self.qEst[action] += self.stepSize * (reward - self.qEst[action])
		return reward


bandit = Bandit(kArm=PATH, epsilon=0.4, degree = structure)
for t in range(20):
	action = bandit.getAction()
	reward = bandit.takeAction(action)
	REWARD[action] += reward

A = []
bound = ()
A.append(np.zeros(PATH) + 1)
A.append(np.zeros(PATH) - 1)
for i in range(PATH):
	bound += ((0.2,0.8),)
res = linprog(REWARD, A_ub=A, b_ub=[5,-5], bounds=bound, options={"disp": True})
print(structure)
print(REWARD / -20)
print(res.x)
