# Resting Potential Simulator
**Simulating and Learning Neuronal Resting Potential Dynamics with Physics and AI**

## Publication
[![DOI](https://zenodo.org/badge/1050006043.svg)](https://doi.org/10.5281/zenodo.17211127)

## 📌 Overview
This project combines **biophysical modeling** and **deep learning** to simulate and predict the resting membrane potential of neurons.  
We start with a **deterministic model** based on ion gradients and membrane conductances, then extend it with an **LSTM neural network** implemented in PyTorch to learn the voltage dynamics directly from data.

The goal is to create a **hybrid neuroscience + AI framework** that can:  
- Accurately simulate membrane potential dynamics.  
- Learn neuronal behavior from generated or experimental data.  
- Compare physics-based and data-driven approaches.  

## ✨ Features
- Biophysical resting potential model (leak currents + Na⁺/K⁺ pump)  
- Calculation of reversal potentials using **Nernst** and **GHK** equations  
- Time integration using **SciPy ODE solvers**  
- Dataset generation for machine learning  
- LSTM implementation with **PyTorch**  
- Visualization and comparison of physical vs learned dynamics  
- Modular structure for extensions (e.g., active channels, multi-compartment models)  
- Experiment results saved in CSV and figures generated for analysis  

## 🧠 Scientific Background
The **resting membrane potential** of a neuron is the stable voltage across its membrane at rest, typically around **-70 mV**.  
It results from:
- Ion concentration gradients (K⁺, Na⁺, Cl⁻)  
- Selective membrane permeability  
- Electrogenic pumps (Na⁺/K⁺ ATPase)  

We model:
- **Leak currents** for K⁺, Na⁺, and Cl⁻  
- Membrane capacitance  
- **Nernst equation** for reversal potentials  
- **GHK equation** for resting potential estimate  

Later, we train an **LSTM neural network** to learn this dynamic behavior from data.

## 📘 Usage
1) Generate Dataset for AI
```
python -m dataset.data_generator
```

2) Train the LSTM
```
python -m ml.train
```

3) Trajectory prediction and evaluation
```
python -m ml.evaluate
```

4) Compare Models and Generate Figures
```
python -m ml.train_experiments
```


## 📊 Example Results
- Physical model: V_rest ≈ -70 mV under standard conditions  
- LSTM: RMSE < 1 mV after training  
- Speed-up: LSTM predicts trajectories faster than SciPy ODE solver  
- Experiment results saved in `experiment_results.csv`  
- Figures generated for visualization of dynamics and comparison

