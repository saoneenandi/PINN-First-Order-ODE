# Physics-Informed Neural Network (PINN) for ODEs

## Overview

This repository implements a **Physics-Informed Neural Network (PINN)** to solve ordinary differential equations (ODEs) using automatic differentiation in PyTorch.

Unlike classical numerical solvers, PINNs embed the governing differential equation directly into the loss function, enabling the model to learn solutions that satisfy both **data constraints** and **physical laws**.

The current implementation focuses on a first-order ODE and serves as a minimal, extensible foundation for more advanced systems such as nonlinear ODEs and the Lane–Emden equation.

---

## Interactive Demo (Google Colab)

Run the PINN implementation directly in your browser:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jHSVnPwBPpqbP_kFYT_ggqeLhdTJPTv1)

The notebook demonstrates:
- End-to-end PINN training  
- Automatic differentiation for ODE constraints  
- Comparison with analytical solution  

---

## Problem Statement

We consider the initial value problem (IVP):

\[
\frac{du}{dt} = \cos(2\pi t), \quad t \in [0,2], \quad u(0) = 1
\]

This problem admits a unique classical solution due to the smoothness of the right-hand side.

The analytical solution is:

\[
u(t) = \frac{\sin(2\pi t)}{2\pi} + 1
\]

The objective is to train a neural network \( u_\theta(t) \) that satisfies the differential equation and initial condition without explicitly using solution data.

---

## Methodology

### Physics-Informed Loss

#### Physics Loss (ODE residual)

\[
\mathcal{L}_{physics} = \left| \frac{du_\theta}{dt} - \cos(2\pi t) \right|^2
\]

#### Initial Condition Loss

\[
\mathcal{L}_{IC} = \left| u_\theta(0) - 1 \right|^2
\]

#### Total Loss

\[
\mathcal{L} = \mathcal{L}_{physics} + \mathcal{L}_{IC}
\]

Derivatives \( \frac{du_\theta}{dt} \) are computed using PyTorch’s automatic differentiation (`torch.autograd`), enabling exact differentiation of the neural network with respect to inputs.

---

## Model Architecture

A fully-connected neural network is used:

- **Input:** \( t \in \mathbb{R} \)  
- **Hidden layers:** 3 layers × 50 neurons  
- **Activation:** Tanh  
- **Output:** \( u(t) \)  

This architecture is well-suited for smooth function approximation and is commonly used in PINN literature.

---

## Repository Structure

```
PINN-First-Order-ODE/
│
├── src/
│   ├── model.py        # Neural network definition
│   ├── loss.py         # Physics-informed loss
│   ├── train.py        # Training loop
│   ├── utils.py        # Plotting and evaluation
│
├── outputs/
│   ├── plots/          # Loss curves and predictions
│   ├── models/         # Saved model checkpoints
│
├── experiments/
│   └── ode_system1/    # Experiment configs and results
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Results

The trained PINN successfully recovers the analytical solution over the domain \( t \in [0,2] \).

### Observations

- Accurate function approximation across the domain  
- Stable convergence of both physics and initial condition losses  
- Log-scale loss curves demonstrate consistent optimization behavior  

Example outputs include:

- Training loss decomposition (physics vs IC)  
- Predicted vs analytical solution plots  

---

## Installation

```bash
git clone https://github.com/saoneenandi/PINN-First-Order-ODE.git
cd PINN-First-Order-ODE
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py
```

Outputs will be saved in the `outputs/` directory.

---

## Key Features

- Autograd-based derivative computation  
- Physics-constrained training  
- Modular and extensible code structure  
- Reproducible experiments  
- Clean separation of components  

---

## Limitations

- Only first-order ODEs  
- No adaptive loss weighting  
- Uniform sampling  

---

## Future Work

- Nonlinear and higher-order ODEs  
- Lane–Emden equation  
- Singularity handling  
- Eigenvalue problems  
- Advanced architectures (SIREN, Fourier features)  

---

## References

Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).  
*Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.*  
Journal of Computational Physics.

---

## Acknowledgment

This project was developed as part of an exploration into **Scientific Machine Learning**, focusing on integrating deep learning with differential equation modeling for physics-driven systems.
