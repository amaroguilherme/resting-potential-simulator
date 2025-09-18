import numpy as np
import matplotlib.pyplot as plt

def plot_voltage_trace(t, V, title="Membrane Potential Trace", save_path=None):
    """
    Plot the membrane potential trace over time.

    Parameters
    ----------
    t : ndarray
        Time vector (ms).
    V : ndarray
        Membrane potential values (mV).
    title : str, optional
        Title of the plot.
    save_path : str, optional
        If provided, saves the figure to this path.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(t, V, label="V(t)", color="black")
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def compute_steady_state_voltage(t, V, window=50.0):
    """
    Compute the steady-state voltage by averaging the last 'window' ms of the simulation.

    Parameters
    ----------
    t : ndarray
        Time vector (ms).
    V : ndarray
        Membrane potential values (mV).
    window : float, optional
        Duration (ms) at the end of the simulation to average over.

    Returns
    -------
    V_mean : float
        Mean membrane potential over the window (mV).
    V_std : float
        Standard deviation of the potential over the window (mV).
    """
    t_end = t[-1]
    mask = t >= (t_end - window)

    V_mean = np.mean(V[mask])
    V_std = np.std(V[mask])

    return V_mean, V_std
