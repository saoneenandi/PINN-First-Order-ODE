# Physics-Informed Neural Network (PINN) for ODEs

## Overview

This repository implements a **Physics-Informed Neural Network (PINN)** to solve ordinary differential equations (ODEs) using automatic differentiation in PyTorch.

Unlike classical numerical solvers, PINNs embed the governing differential equation directly into the loss function, enabling the model to learn solutions that satisfy both **data constraints** and **physical laws**.

The current implementation focuses on a first-order ODE and serves as a minimal, extensible foundation for more advanced systems such as nonlinear ODEs and the Lane–Emden equation.

---

## Problem Statement

We consider the differential equation:

[
\frac{du}{dt} = \cos(2\pi t), \quad u(0) = 1
]

The analytical solution is given by:

[
u(t) = \frac{\sin(2\pi t)}{2\pi} + 1
]

The objective is to train a neural network ( u_\theta(t) ) that satisfies the differential equation and initial condition without explicitly using solution data.

---

## Methodology

### Physics-Informed Loss

The model is trained by minimizing a composite loss function:

* **Physics Loss** (ODE residual):
  [
  \mathcal{L}*{physics} = \left| \frac{du*\theta}{dt} - \cos(2\pi t) \right|^2
  ]

* **Initial Condition Loss**:
  [
  \mathcal{L}*{IC} = \left| u*\theta(0) - 1 \right|^2
  ]

* **Total Loss**:
  [
  \mathcal{L} = \mathcal{L}*{physics} + \mathcal{L}*{IC}
  ]

Gradients are computed using PyTorch’s automatic differentiation (`torch.autograd`).

---

## Model Architecture

A fully-connected neural network is used:

* Input: ( t \in \mathbb{R} )
* Hidden layers: 3 layers × 50 neurons
* Activation: Tanh
* Output: ( u(t) )

This architecture is well-suited for smooth function approximation and is commonly used in PINN literature.

---

## Repository Structure

```
pinn-ode-solver/
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

The trained PINN successfully recovers the analytical solution:

* Accurate function approximation across the domain ( t \in [0, 2] )
* Stable convergence of physics and boundary losses
* Log-scale loss curves demonstrate consistent optimization behavior

Example outputs include:

* Training loss decomposition (physics vs IC)
* Predicted vs analytical solution plots

---

## Installation

```bash
git clone https://github.com/your-username/pinn-ode-solver.git
cd pinn-ode-solver
pip install -r requirements.txt
```

---

## Usage

Run training:

```bash
python main.py
```

Outputs (plots and logs) will be saved in the `outputs/` directory.

---

## Key Features

* Autograd-based derivative computation
* Physics-constrained training (no labeled data required)
* Modular and extensible code structure
* Reproducible experiments via fixed random seeds
* Clear separation of model, loss, and training logic

---

## Limitations

* Current implementation handles only first-order ODEs
* No adaptive loss weighting between physics and boundary terms
* Uniform sampling of collocation points (no adaptive refinement)

---

## Future Work

* Extension to second-order and nonlinear ODEs
* Implementation of **Lane–Emden equation** (astrophysical polytropes)
* Handling singularities (e.g., ( \xi = 0 ))
* Eigenvalue problems and stability analysis
* Advanced architectures (Fourier features, SIREN)
* Adaptive sampling and loss balancing

---

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).
   *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.*
   Journal of Computational Physics.

---

## Acknowledgment

This project was developed as part of an exploration into **Scientific Machine Learning** and its applications to differential equations and astrophysical modeling.

---
