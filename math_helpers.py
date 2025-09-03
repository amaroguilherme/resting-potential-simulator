import math

from constants import K_mNa, K_mK

def nernst(out_conc, in_conc, z=1):
    return (61/z) * math.log10(out_conc/in_conc)


def ghk_voltage(permeabilities: dict, out_concs: dict, in_concs: dict):
    avg_permeability = ((permeabilities['K'] * out_concs['K']) + (permeabilities['Na'] * out_concs['Na']) + (permeabilities['Cl'] * in_concs['Cl'])/
                        (permeabilities['K'] + in_concs['K']) + (permeabilities['Na'] * in_concs['Na']) + permeabilities['Cl'] + out_concs['Cl'])
    
    return 61 * (math.log10(avg_permeability))


def michaelis_menten(Na_in, K_out):
    return (Na_in/(Na_in + K_mNa)) * (K_out/(K_out + K_mK))
