import gymnasium as gym
import gym_pusht
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from jepa import Jepa
from torch import nn
from lightly.loss import VICRegLoss

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

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



def cross_entropy_method(jepa, s1, a1, s2, goal, horizon, n_samples, n_elite, n_iters):
    mean = torch.full((horizon, 2), 256, device=device)
    std = torch.full((horizon, 2), 100.0, device=device)

    init = np.concatenate([s1, a1, s2])
    init_t = torch.from_numpy(init).float().to(device)
    z_start = jepa.encoder(init_t.unsqueeze(0)).repeat(n_samples, 1)

    goal_full = np.concatenate([s2, a1, np.array([256., 256., goal[0], goal[1], goal[2]])])
    goal_t = torch.from_numpy(goal_full).float().to(device)
    goal_encoded = jepa.encoder(goal_t.unsqueeze(0))[0]

    for i in range(n_iters):
        actions = std * torch.randn(n_samples, horizon, 2, device=device) + mean
        actions = actions.clamp(0, 512)
        z = z_start.clone()

        with torch.no_grad():
            for t in range(horizon):
                a_t = actions[:, t, :]
                z = jepa.predictor(torch.cat([z, a_t], dim=1))
            costs = ((z[:, -3:] - goal_encoded[-3:]) ** 2).sum(dim=1)

        elites_idx = costs.topk(n_elite, largest=False).indices
        elite_actions = actions[elites_idx]
        mean = elite_actions.mean(dim=0)
        std = elite_actions.std(dim=0) + 1e-6
        # print(i, " mean ", mean[0], " std ", std[0])
    return mean[0]




jepa = Jepa(6, 10, 8, 8)
train_jepa(jepa, 100000, 10000, 500, 1e-3)

env = gym.make("gym_pusht/PushT-v0", render_mode="human")

s1, _ = env.reset()
a1 = env.action_space.sample()
s2, _, _, _, _ = env.step(a1)
goal_state = env.unwrapped.goal_pose


for i in range(200):
    mean = cross_entropy_method(jepa, s1, a1, s2, goal_state, 180, 350, 80, 500)
    s1 = s2
    a1 = mean.cpu().numpy()
    s2, _, _, _, _ = env.step(a1)
    env.render()