# Resting Potential Simulator
**Simulating and Learning Neuronal Resting Potential Dynamics with Physics and AI**

## 📌 Overview
This project combines **biophysical modeling** and **deep learning** to simulate and predict the resting membrane potential of neurons.  
We start with a **deterministic model** based on ion gradients and membrane conductances, then extend it with a **Neural ODE** implemented in PyTorch to learn the voltage dynamics directly from data.

The goal is to create a **hybrid neuroscience + AI framework** that can:  
- Accurately simulate membrane potential dynamics.  
- Learn neuronal behavior from generated or experimental data.  
- Compare physics-based and data-driven approaches.  

## ✨ Features
- Biophysical resting potential model (leak currents + Na⁺/K⁺ pump)  
- Calculation of reversal potentials using **Nernst** and **GHK** equations  
- Time integration using **SciPy ODE solvers**  
- Dataset generation for machine learning  
- Neural ODE implementation with **PyTorch + torchdiffeq**  
- Visualization and comparison of physical vs learned dynamics  
- Modular structure for extensions (e.g., active channels, multi-compartment models)  

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

Later, we train a **Neural ODE** to learn this dynamic behavior from data.

## 📘 Usage
1) Generate Dataset for AI
```
python src/data_generator.py --n_samples 1000 --output data/dataset.npz
```

2) Train the Neural ODE
```
python src/experiments/train_neural_ode.py --config params/training_config.yml
```

3) Compare Models
```
python src/experiments/compare_vs_physical.py
```

4) Run the Biophysical Model (optional)
```
python src/experiments/run_physical_model.py
```

## 📊 Example Results
- Physical model: V_rest ≈ -70 mV under standard conditions
- Neural ODE: RMSE < 1 mV after training
- Speed-up: Neural ODE predicts trajectories faster than SciPy ODE solver