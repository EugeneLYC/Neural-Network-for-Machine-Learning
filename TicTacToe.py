'''
    This project originates from chapter 1 in Reinforcement Learning, named TicTacToe.
    2017.4.28
'''

import numpy as np
import pickle

EPOCHS = 200
ROWS = 3
COLS = 3
SIZE = ROWS * COLS

class State:
    def __init__(self):
        self.data = np.zeros((ROWS,COLS))
        self.winner = None
        self.hashVal = None
        self.end = None
        
    def getHash(self):
        if self.hashVal is None:
            self.hashVal = 0
            for i in self.data.reshape(ROWS * COLS):
                if i == -1:
                    i = 2
                self.hashVal = self.hashVal * 3 + i
        return int(self.hashVal)
    
    def isEnd(self):
        if self.end is not None:
            return self.end
        results = []
        for i in range(0,ROWS):
            results.append(np.sum(self.data[i,:]))
        for i in range(0,COLS):
            results.append(np.sum(self.data[:,i]))
        
        results.append(0)
        for i in range(0,ROWS):
            results[-1] += self.data[i,i]
        results.append(0)
        for i in range(0,ROWS):
            results[-1] += self.data[i,ROWS - 1 - i]
        
        for result in results:
            if result == 3:
                self.winner = 1
                self.end = True
                return self.end
            if result == -3:
                self.winner = -1
                self.end = True
                return self.end
            
        sum = np.sum(np.abs(self.data))
        if sum == ROWS * COLS:
            self.winner = 0
            self.end = True
            return self.end
        
        self.end = False
        return self.end
    
    def nextState(self,i,j,symbol):
        newState = State()
        newState.data = np.copy(self.data)
        newState.data[i,j] = symbol
        return newState
    
    def show(self):
        for i in range(0, ROWS):
            print('-------------')
            out = '| '
            for j in range(0, COLS):
                if self.data[i, j] == 1:
                    token = '*'
                if self.data[i, j] == 0:
                    token = '0'
                if self.data[i, j] == -1:
                    token = 'x'
                out += token + ' | '
            print(out)
        print('-------------')
        
def getAllStatesImpl(currentState, currentSymbol, allStates):
    for i in range(ROWS):
        for j in range(COLS):
            if currentState.data[i][j] == 0:
                newState = currentState.nextState(i,j,currentSymbol)
                newHash = newState.getHash()
                if newHash not in allStates.keys():
                    isEnd = newState.isEnd()
                    allStates[newHash] = (newState, isEnd)
                    if not isEnd:
                        getAllStatesImpl(newState, -currentSymbol, allStates)
                        
def getAllStates():
    currentSymbol = 1
    currentState = State()
    allStates = dict()
    allStates[currentState.getHash()] = (currentState, currentState.isEnd())
    getAllStatesImpl(currentState, currentSymbol, allStates)
    return allStates
    
allStates = getAllStates()

class Judger:
    def __init__(self, player1, player2, feedback = True):
        self.p1 = player1
        self.p2 = player2
        self.feedback = feedback
        self.currentPlayer = None
        self.p1Symbol = 1
        self.p2Symbol = -1
        self.p1.setSymbol(self.p1Symbol)
        self.p2.setSymbol(self.p2Symbol)
        self.currentState = State()
        self.allStates = allStates
        
    def giveReward(self):
        if self.currentState.winner == self.p1Symbol:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif self.currentState.winner == self.p2Symbol:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0)
            self.p2.feedReward(0)
    
    def feedCurrentState(self):
        self.p1.feedState(self.currentState)
        self.p2.feedState(self.currentState)

    def reset(self):
        self.p1.reset()
        self.p2.reset()
        self.currentState = State()
        self.currentPlayer = None
        
    def play(self, show = False):
        self.reset()
        self.feedCurrentState()
        #enter the loop of changing the players taking the move
        while True:
            if self.currentPlayer == self.p1:
                self.currentPlayer = self.p2
            else:
                self.currentPlayer = self.p1
            if show:
                self.currentState.show()
            [i, j, symbol] = self.currentPlayer.takeAction()
            self.currentState = self.currentState.nextState(i, j, symbol)
            hashValue = self.currentState.getHash()
            self.currentState, isEnd = self.allStates[hashValue]
            self.feedCurrentState()
            if isEnd:
                if self.feedback:
                    self.giveReward()
                return self.currentState.winner                
        
    
#AI Player
class Player:
    def __init__(self, stepSize = 0.1, exploreRate = 0.1):
        self.allStates = allStates
        self.estimations = dict()
        self.stepSize = stepSize
        self.exploreRate = exploreRate
        self.states = []
        
    def reset(self):
        self.states = []
        
    def setSymbol(self, symbol):
        self.symbol = symbol
        '''
            Every player makes moves based on their estimations.
            In this function, we set winning terminal as 1.0, losing state as 0 and the rest as 0.5
        '''
        for hash in allStates.keys():
            (state, isEnd) = allStates[hash]
            if isEnd:
                if state.winner == self.symbol:
                    self.estimations[hash] = 1.0
                else:
                    self.estimations[hash] = 0
            else:
                self.estimations[hash] = 0.5
                
    def feedState(self, state):
        self.states.append(state)
    
    def feedReward(self,reward):
        '''
            This function updates estimations according to the reward
        '''                        
        if len(self.states) == 0:
            return
        self.states = [state.getHash() for state in self.states]
        target = reward
        for latestState in reversed(self.states):
            value = self.estimations[latestState] + self.stepSize * (target - self.estimations[latestState])
            self.estimations[latestState] = value
            target = value
        self.states = []
        
    def takeAction(self):
        state = self.states[-1]
        nextStates = []
        nextPositions = []
        for i in range(ROWS):
            for j in range(COLS):
                if state.data[i,j] == 0:
                    nextPositions.append([i,j])
                    nextStates.append(state.nextState(i, j, self.symbol).getHash())
        if np.random.binomial(1, self.exploreRate):
            np.random.shuffle(nextPositions)
            self.states = []
            action = nextPositions[0]
            action.append(self.symbol)
            return action
        
        values = []
        for hash, pos in zip(nextStates, nextPositions):
            values.append((self.estimations[hash], pos))
        np.random.shuffle(values)
        values.sort(key = lambda x: x[0], reverse = True)
        action = values[0][1]
        action.append(self.symbol)
        return action
    
    def savePolicy(self):
        fw = open('optimal_policy_' + str(self.symbol), 'wb')
        pickle.dump(self.estimations, fw)
        fw.close()

    def loadPolicy(self):
        fr = open('optimal_policy_' + str(self.symbol),'rb')
        self.estimations = pickle.load(fr)
        fr.close()


# human interface
# input a number to put a chessman
# | 1 | 2 | 3 |
# | 4 | 5 | 6 |
# | 7 | 8 | 9 |
class humanPlayer:
    def __init__(self, stepSize = 0.1, exploreRate = 0.1):
        self.symbol = None
        self.currentState = None
        
    def reset(self):
        return
    
    def setSymbol(self, symbol):
        self.symbol = symbol
        return
    
    def feedState(self, state):
        self.currentState = state
        return
    
    def feedReward(self, reward):
        return
    
    def takeAction(self):
        data = int(input("Input your position:"))
        data -= 1
        i = data // int(COLS)
        j = data % COLS
        if self.currentState.data[i, j] != 0:
            return self.takeAction()
        return (i, j, self.symbol)

def train(EPOCHS = 200):
    player1 = Player()
    player2 = Player()
    judger = Judger(player1, player2)
    player1Win = 0.0
    player2Win = 0.0
    for i in range(0, epochs):
        print("Epoch", i)
        winner = judger.play()
        if winner == 1:
            player1Win += 1
        if winner == -1:
            player2Win += 1
        judger.reset()
    print(player1Win / epochs)
    print(player2Win / epochs)
    player1.savePolicy()
    player2.savePolicy()
    
def compete(turns=500):
    player1 = Player(exploreRate=0)
    player2 = Player(exploreRate=0)
    judger = Judger(player1, player2, False)
    player1.loadPolicy()
    player2.loadPolicy()
    player1Win = 0.0
    player2Win = 0.0
    for i in range(0, turns):
        print("Epoch", i)
        winner = judger.play()
        if winner == 1:
            player1Win += 1
        if winner == -1:
            player2Win += 1
        judger.reset()
    print(player1Win / turns)
    print(player2Win / turns)

def play():
    while True:
        player1 = Player(exploreRate=0)
        player2 = humanPlayer()
        judger = Judger(player1, player2, False)
        player1.loadPolicy()
        winner = judger.play(True)
        if winner == player2.symbol:
            print("Win!")
        elif winner == player1.symbol:
            print("Lose!")
        else:
            print("Tie!")

train()
compete()
play()
