import random
from queue import PriorityQueue
from random import sample 
from collections import OrderedDict
import time
import numpy as np
import torch

from src.utils.utils import read_instances

from src.dd.solution import Solution
from src.dd.state import DDState, createChildDDState
from src.dd.searchStrategy import DepthFirst, DepthFirstWithRerun, Random, BestFirst, Balanced

# from src.rl.environment import ShopEnv, ShopGraphEnv

class Solver:
    def __init__(self,
                input,):
        
        # TO DO: confirm copy behaviour of these 
        self.problem = read_instances(input,full=True)
        self.precedence = self.problem.precedence
        self.processing = self.problem.processing
        self.solution = Solution()
        self.states = PriorityQueue()
        self.currentState = None
        # self.searchStrategy = DepthFirst()
        self.searchStrategy = Balanced()
        # self.searchStrategy = BestFirst()
        self.undominatedStates = {} # Dict of operation status and state Dict(Dict(OperationId,Bool),Dict(StateId,State))
        # keep the vertexId
        self.vertexId = 0    

    def solveWithRerun(self,args,isAgent=False,agentType=None,model=None,env=None):
        # problemInstance = self.problem
        solution = self.solution

        # initialise state queue and search strategy
        allStates = PriorityQueue() # defaults to ascending order so smallest value has highest priority
        
        # allSearchStrategy = Random()
        allSearchStrategy = BestFirst()
        # allSearchStrategy = Balanced()
        deepSearchStrategy = DepthFirst()

        # create root state and add to queue
        root = DDState(id = self.vertexId,
                depth = 0,
                state = {
                        'machine_utilization': np.zeros((self.problem.nops,), dtype=int),
                        'job_times': self.processing,
                        'job_early_start_time': np.zeros((self.problem.njobs,), dtype=int),
                        'precedence': self.precedence,
                        'job_state': np.zeros(self.problem.njobs,dtype=np.int64) ,
                        'jobs_to_process': self.problem.njobs,
                        'max_span': 0.0,
                        'placement': np.zeros((self.problem.nops,self.problem.njobs,2), dtype=int),
                        'order':np.zeros((self.problem.nops), dtype=int),
                        'job_done':np.zeros(self.problem.njobs)
                    },
                feasibleSet=[j for j in range(self.problem.njobs)])

        allStates.put((allSearchStrategy.getRanking(root,rankFactor=0.5,totalOps=self.problem.njobs*self.problem.nops,bestUpperBound=solution.upperBound),root)) #push tuple of priority and value
        
        iterationCount = 0
        startTime = time.time()
        ended = False
        while not allStates.empty():
            if ended:
                break
            #create deep state queue and add 
            deepStates = PriorityQueue() #trying the depth first + rerun idea
            prio, aState = allStates.get()

            if aState.isDominated:
                continue
            
            deepStates.put((deepSearchStrategy.getRanking(aState,rankFactor=0.5,totalOps=self.problem.njobs*self.problem.nops,bestUpperBound=solution.upperBound),aState)) #push tuple of priority and value
            print(f'Iteration {iterationCount}')
            while not deepStates.empty():
                
                # log.debug(f'All states has {allStates.qsize()} elements at loop top')
                solveTime = time.time() - startTime
                if solution.isOptimal or solveTime >= args.timeout or iterationCount >= args.iterations:
                    while not deepStates.empty():
                        prio, dState = deepStates.get()
                        allStates.put((allSearchStrategy.getRanking(dState,rankFactor=0.5,totalOps=self.problem.njobs*self.problem.nops,bestUpperBound=solution.upperBound),dState)) #push tuple of priority and value
                    # log.debug(f'All states has {allStates.qsize()} elements at end condition')
                    ended = True
                    break
                
                # Remove and return the highest priority element
                currentPrio, self.currentState = deepStates.get()

                if self.currentState.isDominated:
                    continue

                # log.debug(f'Current top priority {currentPrio} of state {self.currentState.id}')

                if self.currentState.isTerminal:
                    solution.addNewSolution(self.currentState)
                    print("Found a solution, update all states")

                    # reached a feasible end of deep dive
                    # add all elements to the original all states queue
                    # log.info(f'length of deep states is {deepStates.qsize()}')
                    while not deepStates.empty():
                        prio, dState = deepStates.get()
                        allSearchStrategy.getRanking(dState,rankFactor=0.5,totalOps=self.problem.njobs*self.problem.nops,bestUpperBound=solution.upperBound)
                        allStates.put((allSearchStrategy.getRanking(dState,rankFactor=0.5,totalOps=self.problem.njobs*self.problem.nops,bestUpperBound=solution.upperBound),dState)) #push tuple of priority and value
                    break

                # print(f"feasible actions are {self.currentState.feasibleSet}")
                if isAgent:
                    pass
                else:
                    chosenJob = np.random.choice(self.currentState.feasibleSet)
                    # print(f"feasible actions are {chosenJob}")
                self.currentState.feasibleSet = np.delete(self.currentState.feasibleSet, np.argwhere(self.currentState.feasibleSet==chosenJob))
                if self.currentState.feasibleSet.size != 0:
                    # add the state back to memory
                    # print("Add state back to memory, still to process")
                    deepStates.put((currentPrio,self.currentState))

                # create new state that schedules op after self.currentState

                self.vertexId+=1
                newState = createChildDDState(self.currentState,chosenJob,self.vertexId,self.problem.nops)

                # if not newState.isFeasible:
                #     # reached an infeasible end of deep dive
                #     # add all elements to the original all states queue
                #     log.info(f'State {newState.id} is infeasible')
                #     while not deepStates.empty():
                #         prio, dState = deepStates.get()
                #         allStates.put((allSearchStrategy.getRanking(dState,rankFactor=0.5,totalOps=problemInstance.njobs*problemInstance.nops,bestUpperBound=solution.upperBound),dState)) #push tuple of priority and value
                #     break

                if newState.state['max_span'] > solution.upperBound:
                    print("Worse than current best, restart depth")
                    while not deepStates.empty():
                        prio, dState = deepStates.get()
                        allStates.put((allSearchStrategy.getRanking(dState,rankFactor=0.5,totalOps=self.problem.njobs*self.problem.nops,bestUpperBound=solution.upperBound),dState)) #push tuple of priority and value
                    break
                    # continue

                # if not self.isDominantState(problemInstance,self.undominatedStates,newState):
                #     log.info(f'State {newState.id} is dominated')
                #     continue

                deepStates.put((deepSearchStrategy.getRanking(newState,rankFactor=0.5,totalOps=self.problem.njobs*self.problem.nops,bestUpperBound=solution.upperBound),newState))

            # update best lower bound
            solution.lowerBound = solution.upperBound if allStates.empty() else min(allStates.queue, key = lambda x: x[1].state["max_span"])[1].state["max_span"]#--- inner list of priority queue
            print(f'best lower bound is {solution.lowerBound}')
            print(f'best upper bound is {solution.upperBound}')
                
            iterationCount += 1
        
        return solution
    
    # # TO DO: is undominated states by reference? confirm
    # def isDominantState(self,problem:Instance,undominatedStates:dict,currentState:DDState):
    #     if binList2str(currentState.operationStatus.values()) in undominatedStates.keys():
    #         for undomId,undomState in undominatedStates[binList2str(currentState.operationStatus.values())].items():
    #             if undomState.isDominant(problem,currentState):
    #                 return False

    #         for undomId,undomState in undominatedStates[binList2str(currentState.operationStatus.values())].items():
    #             if currentState.isDominant(problem,undomState):
    #                 undominatedStates[binList2str(currentState.operationStatus.values())].pop(undomId)
    #                 undominatedStates[binList2str(currentState.operationStatus.values())].update({currentState.id:currentState}) 

    #                 undomState.cleanUpPath()
    #                 undomState.isDominated = True

    #     undominatedStates[binList2str(currentState.operationStatus.values())] = {currentState.id:currentState}
    #     return True
    

        


