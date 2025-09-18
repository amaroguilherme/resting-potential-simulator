"""
    Create different scenarios to run the simulation:
    - Baseline: nothing changes, only the resting potential is simulated.
    - Current step: applies an artificially injected current to observe the membrane response.
    - K_out ramp: gradually changes the extracellular potassium concentration, simulating a classic experiment to study how this affects V.

    These functions do not solve the ODE directly; they configure the parameters and call the integrator (run_simulation).
"""
import numpy as np
from biophysics.integrator import run_simulation


def baseline_protocol():
    """
        Simulate the cell under normal conditions, without stimuli. 
        - Defines the default parameters (normal physiological concentrations, conductances, pump, etc.).
        - Uses the integrator (run_simulation) to run for a period of time (e.g., 500 ms).
        - Expects to see the potential stabilize at around ~-70 mV.
    """
    return run_simulation(-70, 0, 500, 0.01)


def current_step_protocol():
    """
        Test the membrane's response to an external current applied for a period.
        During the simulation, an extra current Iinj is injected for a defined period (e.g., 100 ms).
    """
    return run_simulation(-70, 0, 500, 0.01, 
                          t_start_stim=100, t_end_stim=200, I_inj=0.1)
    

def k_out_ramp_protocol(V0=-70.0,
                        K_out_start=5.0,
                        K_out_end=10.0,
                        ramp_start_time=100.0,
                        ramp_end_time=300.0,
                        t_start=0.0,
                        t_end=500.0,
                        dt=0.01,
                        params=None):
    """
        Simulate the gradual change of K_out, as in classic experiments that alter the extracellular bath.

        - During the simulation, K_out is increased linearly (e.g., from 3 mM to 10 mM over 200 ms).
        - At each call of the rhs function, the value of K_out is adjusted based on the current time t.
    """
    t = np.arange(t_start, t_end, dt)

    K_out_trace = np.full_like(t, K_out_start, dtype=float)

    ramp_mask = (t >= ramp_start_time) & (t <= ramp_end_time)
    K_out_trace[ramp_mask] = np.linspace(
        K_out_start, K_out_end, ramp_mask.sum()
    )
    K_out_trace[t > ramp_end_time] = K_out_end

    params = params.copy() if params is not None else {}
    params["K_out_trace"] = K_out_trace

    t, V = run_simulation(V0=V0,
                          params=params,
                          t_start=t_start,
                          t_end=t_end,
                          dt=dt)

    return t, V, K_out_trace