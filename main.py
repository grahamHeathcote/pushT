import gymnasium as gym
import gym_pusht
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from jepa import Jepa
from torch import nn
from lightly.loss import VICRegLoss

# State Action State
#              State Action State
def generate_training_data(samples):
    env = gym.make("gym_pusht/PushT-v0")
    arr = np.empty((12, samples))
    env.reset()
    observation, _, _, _, _ = env.step(env.action_space.sample())
    for i in range(samples):
        arr[:5, i] = observation
        action = env.action_space.sample()
        arr[5:7, i] = action
        observation, reward, terminated, truncated, _ = env.step(action)
        arr[7:12, i] = observation
    X = arr[:, :samples-1]
    Y = arr[:, 1:]
    return torch.from_numpy(X).float(), torch.from_numpy(Y).float()


def train_jepa(jepa, sample_size, batch_size, epochs, lr=1e-3):
    X_train, Y_train = generate_training_data(sample_size)
    train_ds = TensorDataset(X_train.T, Y_train.T)
    train_data_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    jepa.to(device)

    loss_func = VICRegLoss(
        lambda_param=25.0,
        mu_param=25.0,
        nu_param=1.0,
        eps=1e-4,
    )
    optimizer = torch.optim.Adam(jepa.parameters(), lr)

    for epoch in range(epochs):
        jepa.train()
        loss_sum = 0
        for x, y in train_data_loader:
            a = y[:, 5:7].to(device)
            x = x.to(device)
            y = y.to(device)
            x_encoded =jepa.encoder(x)
            x_encoded_a = torch.cat([x_encoded, a], dim=1)
            y_pred = jepa.predictor(x_encoded_a)
            y_true = jepa.encoder(y)
            loss = loss_func(y_true, y_pred)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_sum += loss.item()
        print(loss_sum)


def cross_entropy_method(jepa, init, goal, horizon, n_samples, n_elite, n_iters, device):
    mean = torch.full((horizon, 2), 256, device=device)
    std = torch.zeros(horizon, 2, device=device)

    for i in range(n_iters):
        actions = std * torch.randn(n_samples, horizon, 2, device=device) + mean
        actions = actions.clamp(0, 512)
        costs = torch.zeros(n_samples, device=device)
        z = init.unsqueeze(0).repeat(n_samples, 1)
        with torch.no_grad():
            for t in range(horizon):
                a_t = actions[:, t, :]
                z = jepa.predictor(torch.cat[z, a_t], dim=1)
            costs = ((x - goal) ** 2).sum(dim =1)

        elites_idx = costs.topk(n_elite, largest=False).indices
        elite_actions = actions[elites_idx]
        mean = elite_actions.mean(dim=0)
        std = elite_actions.std(dim=0) + 1e-6
    return mean[0]

jepa = Jepa(6, 10, 8, 8)
train_jepa(jepa, 50000, 10000, 500, 1e-3)