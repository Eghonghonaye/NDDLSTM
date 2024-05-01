from src.utils.limits import(
    minint,
    maxint,
)

class Solution:
    def __init__(self,
                 lowerBound = None,
                 upperBound = None,
                 solutions = None,
                 bestSolution = None,
                 isOptimal = None,
                 isTimeOut = None,):
        
        if lowerBound is None:
            lowerBound = minint
        if upperBound is None:
            upperBound = maxint
        if solutions is None:
            solutions = []
        if bestSolution is None:
            bestSolution = None
        if isOptimal is None:
            isOptimal = False
        if isTimeOut is None:
            isTimeOut = False

        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.solutions = solutions
        self.bestSolution = bestSolution
        self.isOptimal = isOptimal
        self.isTimeOut = isTimeOut

    def addNewSolution(self,state):
        self.solutions.append(state)
        stateMakespan = state.state["max_span"] #makespan is the lower bound when solution is complete
        if not self.bestSolution or stateMakespan <= self.bestSolution.state["max_span"]:
            self.bestSolution = state
            self.upperBound = stateMakespan

        if stateMakespan <= self.lowerBound:
            self.isOptimal = True

        if stateMakespan < self.upperBound:
            self.upperBound = stateMakespan

    def setLowerBound(self,newBound):
        self.lowerBound = newBound

    def setUpperBound(self,newBound):
        self.upperBound = newBound

    def printBestSolution(self):
        if self.bestSolution:
            self.bestSolution.printAttributes()
        else:
            print("No solution found")