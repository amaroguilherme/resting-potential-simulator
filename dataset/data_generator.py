import os
import numpy as np
from biophysics.integrator import run_simulation
from dataset.parameter_ranges import PARAM_RANGES_DICT

N_samples = 1000
t_start = 0.0
t_end = 500.0
dt = 0.5
seed = 42
save_path = "dataset/files/dataset.npz"

os.makedirs(os.path.dirname(save_path), exist_ok=True)

np.random.seed(seed)

param_names = list(PARAM_RANGES_DICT.keys())
N_parameters = len(param_names)

t = np.arange(t_start, t_end, dt)
N_timepoints = len(t)

X_array = np.zeros((N_samples, N_parameters))
Y_array = np.zeros((N_samples, N_timepoints))

for i in range(N_samples):
    X_i = np.zeros(N_parameters)
    for j, param in enumerate(param_names):
        p_min, p_max, dist_type = PARAM_RANGES_DICT[param]
        if dist_type == "uniform":
            X_i[j] = np.random.uniform(p_min, p_max)
        elif dist_type == "log-uniform":
            X_i[j] = np.exp(np.random.uniform(np.log(p_min), np.log(p_max)))
        elif dist_type == "fixed":
            X_i[j] = p_min
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")
    
    X_array[i, :] = X_i

    V0 = -70.0

    V_i = run_simulation(V0, X_i, t_start, t_end, dt)

    if np.any(np.isnan(V_i)):
        raise ValueError(f"NaN detected in simulation for sample {i}")

    Y_array[i, :] = V_i

    if (i+1) % 50 == 0:
        print(f"{i+1}/{N_samples} samples generated.")

np.savez_compressed(
    save_path,
    X=X_array,
    Y=Y_array,
    t=t,
    param_names=np.array(param_names)
)

print(f"\nDataset saved to {save_path}: {N_samples} samples, {N_timepoints} timepoints each.")
