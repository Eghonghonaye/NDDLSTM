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
        self.undominatedStates = {} # Dict of operation status and state Dict(array(jobstate),Dict(StateId,State))
        # keep the vertexId
        self.vertexId = 0    

    def solveWithRerun(self,args,isAgent=False,model=None):
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
                feasibleSet=[j for j in range(self.problem.njobs)],
                lowerBound=0)

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
                    # create state and normalise with max time
                    rlstate = {
                            'machine_utilization': np.expand_dims(self.currentState.state['machine_utilization']/100, axis=0),
                            'job_times': np.expand_dims(self.currentState.state['job_times']/100, axis=0),
                            'job_early_start_time': np.expand_dims(self.currentState.state['job_early_start_time']/100, axis=0),
                            'precedence': np.expand_dims(self.currentState.state['precedence'], axis=0),
                            'job_state': np.expand_dims(self.currentState.state['job_state'], axis=0)
                        }
                    
                    # apply mask
                    invalid_mask = np.ones(self.problem.njobs)
                    np.put(invalid_mask, self.currentState.feasibleSet,0)
                    # print(f"invalid mask {invalid_mask}, feasible set {self.currentState.feasibleSet}")

                    # Compute action
                    with torch.set_grad_enabled(False): # False as not training
                        actorJobEmb = model["actor"].instance_embed(rlstate)
                        criticJobEmb = model["critic"].instance_embed(rlstate)

                        prob, log_prob = model["actor"](rlstate,actorJobEmb,1,invalid_mask)
                        value = model["critic"](rlstate,criticJobEmb)

                    # argmax from probs_product
                    # probs_product[0].shape = size_seach
                    # probs_product[1].shape = num actions
                    
                    # probs_product = probs_product * prob.cpu() 
                    probs_product = prob.cpu() 
                    argmax_prob = np.dstack(np.unravel_index(np.argsort(probs_product.ravel()), probs_product.shape))[0][::-1]

                    # call forward to get action
                    chosenJob = [argmax_prob[0, 1]][0]

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

                if newState.state['max_span'] >= solution.upperBound or \
                            newState.lowerBound >= solution.upperBound:
                    # print(f"Worse than current best at {newState.state['max_span']} with lowerbound {newState.lowerBound}, restart depth")
                    while not deepStates.empty():
                        prio, dState = deepStates.get()
                        allStates.put((allSearchStrategy.getRanking(dState,rankFactor=0.5,totalOps=self.problem.njobs*self.problem.nops,bestUpperBound=solution.upperBound),dState)) #push tuple of priority and value
                    break
                    # continue

                if not self.isDominantState(newState):
                    # print(f'State {newState.id} is dominated')
                    continue

                deepStates.put((deepSearchStrategy.getRanking(newState,rankFactor=0.5,totalOps=self.problem.njobs*self.problem.nops,bestUpperBound=solution.upperBound),newState))

            # update best lower bound
            solution.lowerBound = solution.upperBound if allStates.empty() else min(allStates.queue, key = lambda x: x[1].lowerBound)[1].lowerBound#--- inner list of priority queue
            print(f'best lower bound is {solution.lowerBound}')
            print(f'best upper bound is {solution.upperBound}')
                
            iterationCount += 1
        
        return solution
    
    # TO DO: is undominated states by reference? confirm
    def isDominantState(self,currentState:DDState):

        jobStateKey = '-'.join([str(val) for val in currentState.state["job_state"]])

        if jobStateKey in self.undominatedStates.keys():
            for undomId,undomState in self.undominatedStates[jobStateKey].items():
                if undomState.isDominant(currentState):
                    # print(f'State {currentState.id} is dominated by {undomId}')
                    return False

            for undomId,undomState in list(self.undominatedStates[jobStateKey].items()):
                if currentState.isDominant(undomState):
                    self.undominatedStates[jobStateKey].pop(undomId)
                    self.undominatedStates[jobStateKey].update({currentState.id:currentState}) 
                    undomState.isDominated = True

        self.undominatedStates[jobStateKey] = {currentState.id:currentState}
        return True
    

        


