from typing import NewType, Dict, Tuple
import numpy as np
from dataclasses import dataclass, field
import sys
import copy

State = NewType("State",Dict[str, np.array])

StateId = NewType("StateId", int)

@dataclass
class DDState:
    state: State = field(default_factory=dict)
    id: StateId = -1
    depth: int = -1
    isDominated: bool = False
    isTerminal: bool = False
    feasibleSet: list = field(default_factory=list)
    lowerBound: int = 0


    def __eq__(self,other):
        return self.id == other.id
    def __lt__(self,other):
        return self.id < other.id
    def __gt__(self,other):
        return self.id > other.id
    
    def isDominant(self,otherState):
        if any(self.state["job_state"] != otherState.state["job_state"]):
            print("wrong dominance comparison")
            sys.exit()
            return True
        
        if all(self.state["machine_utilization"] <= otherState.state["machine_utilization"]):
            # print(self.state["job_state"])
            # print(otherState.state["job_state"])
            # print("yes I am dominant")
            # print(self.state["machine_utilization"])
            # print(otherState.state["machine_utilization"])
            return True
        
        return False
        

    


# State = {
#                 'machine_utilization': machine_utilization,
#                 'job_times': job_times,
#                 'job_early_start_time': job_early_start_time,
#                 'precedence': precedence,
#                 'job_state': jobsState,
#                 'jobs_to_process': jobsToProcess,
#                 'max_span': maxSpan,
#                 'placement': placement,
#                 'order':order,
#                 'job_done':jobDone
#             }

def createChildDDState(ddState:DDState,jobToChoose,vertexId,nops):
    id = vertexId
    depth = ddState.depth + 1
    newstate = createChildState(ddState.state,jobToChoose,nops)
    isDominated = ddState.isDominated
    isTerminal = False if newstate["jobs_to_process"] > 0 else True
    feasibleSet = (newstate['job_done']<=0)*1
    feasibleSet = feasibleSet.nonzero()[0]
    lowerBound = max([newstate["machine_utilization"][m] + 
                      sum([newstate["job_times"][i][j] for i in range(newstate["job_times"].shape[0]) 
                                        for j in range(newstate["job_times"].shape[1]) 
                                        if newstate["precedence"][i][j] == m and newstate["job_state"][i] <= j ])
                                                               for m in range(nops) ])
    # print(f'job state is {newstate["job_state"]}')
    # print(f'job times is {newstate["job_times"]}')
    # print(f'precedence is {newstate["precedence"]}')
    # print(f"lowerbound is {lowerBound}")

    return DDState(
                state = newstate,
                id = id,
                depth = depth,
                isDominated = isDominated,
                isTerminal = isTerminal,
                feasibleSet=feasibleSet,
                lowerBound=lowerBound,
            )

def createChildState(state:State,jobToChoose,maxJobLength,):
    
    job_times = copy.deepcopy(state['job_times'])
    jobsState = copy.deepcopy(state['job_state'])
    smaxSpan = copy.deepcopy(state['max_span'])
    job_early_start_time = copy.deepcopy(state['job_early_start_time'])
    precedence = copy.deepcopy(state['precedence'])
    machine_utilization = copy.deepcopy(state['machine_utilization'])
    jobDone = copy.deepcopy(state['job_done'])
    jobsToProcess = copy.deepcopy(state['jobs_to_process'])
    placement = copy.deepcopy(state['placement'])
    order = copy.deepcopy(state['order'])

    # operation index
    jobOrder = jobsState[jobToChoose] # note: maximum = num_ops
    # print (jobOrder)
    maxSpan = smaxSpan # current truncated makespan
    # if index is out of bound, job was already finished.
    if jobOrder ==  maxJobLength:  
        reward = -1  # masking invalid action would not trigger this if statement
        print(f"chosen invalid action {jobToChoose} in state {state}")
        pass
    else:        
        job_early_start =  job_early_start_time[jobToChoose] # how long does it take to start the operations for the chosen job
        job_time =  job_times[jobToChoose][jobOrder] # operation time for the chosen job
        job_machine_placement =  precedence[jobToChoose][jobOrder] # which machine to perform the operation for the chosen job
        
        
        placement[job_machine_placement,order[job_machine_placement],:] =([jobToChoose, jobOrder])
        order[job_machine_placement] += 1

        machine_current_utilization = machine_utilization[job_machine_placement] # how long this machine has been running

        if job_early_start > machine_current_utilization:
            machine_current_utilization = job_early_start + job_time
        else:
            machine_current_utilization +=  job_time

        job_early_start_time[jobToChoose] = machine_current_utilization   

        machine_utilization[job_machine_placement] = machine_current_utilization
        
        if machine_utilization[job_machine_placement] > maxSpan:
            maxSpan = machine_utilization[job_machine_placement]

        jobsState[jobToChoose]+=1
        if  jobOrder+1 == maxJobLength:
            jobsToProcess -= 1
            jobDone[jobToChoose] = 1 # doesn't really use this?
            
        reward = float(smaxSpan - maxSpan)

    done = 1
    if  jobsToProcess > 0:
        done = 0
    # return jobsToProcess, maxSpan, reward, done, job_early_start_time, machine_utilization, jobsState, jobDone, placement
    newState = {
        'machine_utilization': machine_utilization,
        'job_times': job_times,
        'job_early_start_time': job_early_start_time,
        'precedence': precedence,
        'job_state': jobsState,
        'jobs_to_process': jobsToProcess,
        'max_span': maxSpan,
        'placement': placement,
        'order':order,
        'job_done':jobDone
    }
    return newState


