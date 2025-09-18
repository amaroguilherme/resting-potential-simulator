"""
This file will gather all the functions that calculate the electrical currents across the membrane. 
Each current contributes to the variation of the membrane potential (dV/dt) in the biophysical model.
"""
from helpers.math_helpers import michaelis_menten


def leak_current(V, g_ion, E_ion):
    """
        These are passive currents that flow through leak channels (not activated by voltage or ligands).
        
        V: current membrane potential
        g_ion: leak channel conductance for the ion
        E_ion: Nernst potential of the ion
    """
    return g_ion * (V - E_ion)


def pump_current(Na_in, K_out, I_max=2):
    """
        Represents the action of the Na⁺/K⁺ ATPase pump, which actively transports ions:
        - Removes 3 Na⁺ from the cell and brings in 2 K⁺
        - Maintains ionic gradients essential for the resting potential
        
        I_max: Constant that defines the maximum capacity of the pump.
               Can be adjusted (1–5 µA/cm²) so that the pump compensates for the leak currents, maintaining V_rest ≈ -70 mV.
        Na_in:
        K_out:  
    """
    return I_max * michaelis_menten(Na_in, K_out)
