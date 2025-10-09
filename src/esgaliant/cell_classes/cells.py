import numpy as np
import jax.numpy as jnp
from typing import Union, List, Tuple, Dict, Optional
from dataclasses import dataclass

#-----------------------------------------------------------------------------#
# Cell population 
#-----------------------------------------------------------------------------#

class CellState:
    def __init__(self,n_cells:int = 5000):
        self.n_cells = n_cells