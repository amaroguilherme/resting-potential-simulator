"""
Physical constants used for the resting potential model.
All values are in SI units unless specified otherwise.
"""

# Universal gas constant (J/(mol·K))
R = 8.3145  

# Faraday's constant (C/mol)
F = 96485.3329  

# Standard temperature (Kelvin)
T = 310.15  # 37°C, typical physiological temperature

# Elementary charge (C)
e = 1.602176634e-19  

# Membrane surface area (cm²) - default example
DEFAULT_MEMBRANE_AREA = 1.0e-4  

# Membrane capacitance per unit area (F/cm²)
DEFAULT_CM_PER_AREA = 1.0e-6  

# Conversion constants
MILLI = 1e-3
MICRO = 1e-6
NANO  = 1e-9
PICO  = 1e-12

# Pump saturation constants
K_mNa = 10 # Can fluctuate between 10-20mM. Adjust as needed.
K_mK = 1 # Can fluctuate between 1-5mM. Adjust as needed.

# Leak conductances
g_K = 0.36
g_Na = 0.03
g_Cl = 0.03

# ion concentrations
K_in = 140.0    # mM
K_out = 5.0
Na_in = 10.0
Na_out = 145.0
Cl_in = 4.0
Cl_out = 110.0
