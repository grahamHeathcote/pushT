import gymnasium as gym
import gym_pusht
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from neural_network import Network
from torch import nn


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



batch_size = 500
epochs = 10

X_train, Y_train = generate_training_data(10000)
train_ds = TensorDataset(X_train.T, Y_train.T)
train_data_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
model = Network(12, 10, 10, 12)
model.to(device)

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

for epoch in range(epochs):
    model.train()

    for x, y in train_data_loader:
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = loss_func(y, y_pred)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(loss.item())
