
import numpy as np
from dataclasses import dataclass, field

@dataclass
class Instance:
    njobs:int 
    nops:int 
    precedence: np.array 
    duration: np.array 