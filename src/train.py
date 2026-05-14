import torch
import numpy as np

def train(model, optimizer, loss_fn, iterations=5000):
    total_loss_record = []
    physics_loss_record = []
    ic_loss_record = []

    for itr in range(iterations):
        optimizer.zero_grad()

        # Resample every iteration (important!)
        train_t = (np.random.rand(100) * 2).reshape(-1, 1)

        loss, physics_loss, ic_loss = loss_fn(train_t, model)

        loss.backward()
        optimizer.step()

        total_loss_record.append(loss.item())
        physics_loss_record.append(physics_loss.item())
        ic_loss_record.append(ic_loss.item())

        if itr % 1000 == 0:
            print(f"Iteration {itr}: Loss = {loss.item()}")

    return total_loss_record, physics_loss_record, ic_loss_record
