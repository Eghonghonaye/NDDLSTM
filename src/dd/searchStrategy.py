from dataclasses import dataclass, field
from src.utils.limits import(
    minint,
    maxint,
)
import random
from abc import ABC, abstractmethod

@dataclass
class AbstractSearchStrategy(ABC):
    @abstractmethod
    def getRanking(self,state,**kwargs):
        pass


@dataclass
class DepthFirst(AbstractSearchStrategy):
    def getRanking(self,state,**kwargs):
        totalOps = maxint if "totalOps" not in kwargs else kwargs["totalOps"]
        return totalOps - state.depth
    
@dataclass
class DepthFirstWithRerun(AbstractSearchStrategy):
    def getRanking(self,state,**kwargs):
        totalOps = maxint if "totalOps" not in kwargs else kwargs["totalOps"]
        return totalOps - state.depth

@dataclass
class BestFirst(AbstractSearchStrategy):
    def getRanking(self,state,**kwargs):
        return state.state["max_span"]

@dataclass
class Balanced(AbstractSearchStrategy):

    def getRanking(self,state,**kwargs):
        rankFactor = 0 if "rankFactor" not in kwargs else kwargs["rankFactor"]
        totalOps = maxint if "totalOps" not in kwargs else kwargs["totalOps"]
        bestUpperBound = maxint if "bestUpperBound" not in kwargs else kwargs["bestUpperBound"]
        return (rankFactor*((totalOps-state.depth)/totalOps) + (1-rankFactor)*(state.state["max_span"]/bestUpperBound))
    
@dataclass
class Random(AbstractSearchStrategy):
    def getRanking(self,state,**kwargs):
        return random.randint(0,1000)
    
@dataclass
class ML(AbstractSearchStrategy):
    '''receive model and return some node ranking function based on predictions of model'''
    def getRanking():
        pass