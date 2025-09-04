"""
   Responsible for running the time simulation of the biophysical model using the numerical solver. 
   It acts as the layer that connects the mathematical model (rhs) with the time execution, 
   producing the trajectory of the membrane potential V(t). 
"""
import numpy as np

from model import rhs
from scipy.integrate import odeint 

def run_simulation(V0, t_start, t_end, dt,
                   t_start_stim=None, t_end_stim=None, I_inj=0):
    """
        Integrates the entire model over time, transforming the derivative dV/dt (calculated in the rhs function)
        into a complete trajectory of the membrane potential V(t).
        
        V0: Initial membrane potential (in mV)
        t_start: Initial simulation time (in ms)
        t_end: Final simulation time (in ms)
        dt: Sampling step for the time vector (in ms)
    """
    
    t = np.arange(t_start, t_end, dt)
    V_solution = odeint(rhs(V0, t, t_start_stim,
                            t_end_stim, I_inj))
    
    return (t, V_solution.flatten())