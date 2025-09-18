# main.py

from protocols.protocols import baseline_protocol, current_step_protocol, k_out_ramp_protocol
from protocols.analysis import plot_voltage_trace, compute_steady_state_voltage

def main():
    t, V = baseline_protocol()
    # t, V = current_step_protocol()
    # t, V = k_out_ramp_protocol()

    plot_voltage_trace(t, V, title="Baseline Protocol")

    V_mean, V_std = compute_steady_state_voltage(t, V, window=50.0)
    print(f"Steady-state voltage: {V_mean:.2f} mV ± {V_std:.2f} mV")

if __name__ == "__main__":
    main()
