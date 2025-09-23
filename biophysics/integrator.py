import numpy as np
from scipy.integrate import odeint
from biophysics.model import rhs

def run_simulation(V0, X_i, t_start, t_end, dt,
                   t_start_stim=0.0, t_end_stim=0.0, I_inj=0.0):
    """
    Simulates the membrane potential V(t) using a set of bio-physical parameters.

    Parameters:
        V0 (float): Initial membrane potential (mV)
        X_i (np.ndarray): Vector of parameters (g_K, g_Na, g_Cl, K_out, K_in, Na_out, Na_in)
        t_start (float): Start time (ms)
        t_end (float): End time (ms)
        dt (float): Time step (ms)
        t_start_stim (float): Optional start time of injected current (ms)
        t_end_stim (float): Optional end time of injected current (ms)
        I_inj (float): Optional current injected (µA/cm²)

    Returns:
        V (np.ndarray): Membrane potential trajectory over time
    """

    g_K, g_Na, g_Cl, K_out, K_in, Na_out, Na_in = X_i[:7]

    t = np.arange(t_start, t_end, dt)

    V = odeint(
        rhs, V0, t,
        args=(g_K, g_Na, g_Cl, K_out, K_in, Na_out, Na_in,
              t_start_stim, t_end_stim, I_inj)
    ).flatten()

    return V
