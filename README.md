# PINN Solver for Lane–Emden Equation

This project solves the Lane–Emden equation using Physics-Informed Neural Networks (PINNs), including:

- Eigenvalue formulation (polytropic stability)
- Handling stiff regimes (n > 3)
- Boundary condition enforcement
- Comparison with RK4 numerical solver

## Equation

d/dx (x² dy/dx)/x² + y^n = 0

## Features

- Learn solution y(x)
- Learn eigenvalue λ
- Works for stiff polytropes
- RK4 benchmark comparison

## Run

```bash
pip install -r requirements.txt
python main.py
