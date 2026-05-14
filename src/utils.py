import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_loss(total, physics, ic, save_path):
    plt.figure(figsize=(10, 8))
    plt.plot(total, label='Total Loss')
    plt.plot(physics, label='Physics Loss')
    plt.plot(ic, label='IC Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()

def evaluate(model):
    test_t = np.linspace(0, 2, 100).reshape(-1, 1)
    true_u = np.sin(2 * np.pi * test_t) / (2 * np.pi) + 1

    test_t_tensor = torch.tensor(test_t, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        pred_u = model(test_t_tensor).numpy()

    return test_t, true_u, pred_u
