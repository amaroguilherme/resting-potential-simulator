"""
    Define the mathematical model of the neuron, that is, the differential equation 
    that describes how the membrane potential V changes over time.
"""
from biophysics.constants import C_m

def rhs(V, t, t_start_stim=0.0, t_end_stim=0.0, I_inj=0.0):
    """
        Receives the current state and returns the rate of change of V (dV/dt).
        
        V: current membrane potential
        t: time (required by odeint, not used here)
        I_K, I_Na, I_Cl: leak currents for each ion
        I_pump: Na⁺/K⁺ pump current
        C_m: membrane capacitance

        The negative sign indicates that a positive current leaving the cell decreases the membrane potential.
    """
    if not (t_start_stim <= t <= t_end_stim):
        I_inj = 0.0

    dVdt = -(I_inj) / C_m 
    return dVdt
