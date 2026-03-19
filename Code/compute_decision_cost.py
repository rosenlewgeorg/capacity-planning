import numpy as np
from simulate_cost import simulate_cost

def compute_decision_cost(w, params):
    """
    Wrapper around simulate_cost for arbitrary decisions.
    
    RETURNS the mean total cost produced by the linear decision rule defined by the
    weight vector W under the model parameters specified in the string PARAMS.
    """
    Kmax = params['Kmax']
    w = np.array(w).flatten()
    if len(w) < Kmax + 2:
        print(f"Warning: Weight vector too short; padding with zeros to length {Kmax+2}.")
        w = np.pad(w, (0, Kmax + 2 - len(w)), 'constant')
        
    return simulate_cost(w, params)
