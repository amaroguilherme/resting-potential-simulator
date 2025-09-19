from biophysics.currents import leak_current, pump_current
from biophysics.constants import C_m, Cl_in, Cl_out
from helpers.math_helpers import nernst

def rhs(V, t, g_K, g_Na, g_Cl, K_out, K_in, Na_out, Na_in,
        t_start_stim=0.0, t_end_stim=0.0, I_inj=0.0):
    """
    Computes the derivative dV/dt for the membrane potential including leak currents, 
    Na+/K+ pump, and optional injected current.

    Parameters:
        V (float): current membrane potential (mV)
        t (float): current time (ms)
        g_K, g_Na, g_Cl (float): leak conductances
        K_out, K_in, Na_out, Na_in (float): ionic concentrations
        t_start_stim, t_end_stim (float): interval for injected current (ms)
        I_inj (float): amplitude of injected current (µA/cm²)

    Returns:
        dVdt (float): rate of change of membrane potential
    """

    I_ext = I_inj if t_start_stim <= t <= t_end_stim else 0.0

    E_K = nernst(K_out, K_in)
    E_Na = nernst(Na_out, Na_in)
    E_Cl = nernst(Cl_out, Cl_in)

    I_K = leak_current(V, g_K, E_K)
    I_Na = leak_current(V, g_Na, E_Na)
    I_Cl = leak_current(V, g_Cl, E_Cl)

    I_pump = pump_current(K_in, Na_out)

    dVdt = - (I_K + I_Na + I_Cl + I_pump - I_ext) / C_m

    return dVdt
