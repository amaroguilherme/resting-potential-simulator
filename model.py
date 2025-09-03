"""
    Define the mathematical model of the neuron, that is, the differential equation 
    that describes how the membrane potential V changes over time.
"""
from constants import g_K, g_Na, g_Cl, K_in, K_out, Na_in, Na_out, Cl_in, Cl_out
from currents import leak_current, pump_current
from math_helpers import nernst

def rhs(V, t, C_m):
    """
        Receives the current state and returns the rate of change of V (dV/dt).
        
        V: current membrane potential
        t: time (required by odeint, not used here)
        I_K, I_Na, I_Cl: leak currents for each ion
        I_pump: Na⁺/K⁺ pump current
        C_m: membrane capacitance

        The negative sign indicates that a positive current leaving the cell decreases the membrane potential.
    """
    
    I_K = leak_current(V, g_K, nernst(out_conc=K_out, in_conc=K_in))
    I_Na = leak_current(V, g_Na, nernst(out_conc=Na_out, in_conc=Na_in))
    I_Cl = leak_current(V, g_Cl, nernst(out_conc=Cl_out, in_conc=Cl_in))
    I_pump = pump_current(Na_in, K_out)
    
    return -((I_K + I_Na + I_Cl + I_pump)/C_m)
