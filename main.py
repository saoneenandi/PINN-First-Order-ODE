import torch.optim as optim
from src.model import SimpleNN
from src.loss import ode_system1
from src.train import train
from src.utils import plot_loss

model = SimpleNN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

total, physics, ic = train(model, optimizer, ode_system1)

plot_loss(total, physics, ic, "outputs/plots/loss.png")
