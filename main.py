import gymnasium as gym
import gym_pusht
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from jepa import Jepa
from torch import nn
from lightly.loss import VICRegLoss


STATE_MEAN = torch.tensor([256., 256., 256., 256., np.pi])
STATE_SCALE = torch.tensor([256., 256., 256., 256., np.pi])
ACTION_MEAN = torch.tensor([256., 256.])
ACTION_SCALE = torch.tensor([256., 256.])

def normalize_state(s):
    return (s - STATE_MEAN.to(s.device)) / STATE_SCALE.to(s.device)

def unnormalize_state(s):
    return s * STATE_SCALE.to(s.device) + STATE_MEAN.to(s.device)

def normalize_action(a):
    return (a - ACTION_MEAN.to(a.device)) / ACTION_SCALE.to(a.device)

def unnormalize_action(a):
    return a * ACTION_SCALE.to(a.device) + ACTION_MEAN.to(a.device)

SAS_MEAN = torch.cat([STATE_MEAN, ACTION_MEAN, STATE_MEAN])

SAS_SCALE = torch.cat([STATE_SCALE, ACTION_SCALE, STATE_SCALE])

def normalize_sas(x):
    return (x - SAS_MEAN.to(x.device)) / SAS_SCALE.to(x.device)

def unnormalize_sas(x):
    return x * SAS_SCALE.to(x.device) + SAS_MEAN.to(x.device)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


def behavior_action(obs, rng):
    return np.clip(obs[2:4] + rng.normal(0, 60, 2), 0, 512)

# State Action State
#              State Action State
def generate_training_data(samples, seed=0):
    env = gym.make("gym_pusht/PushT-v0")
    rng = np.random.default_rng(seed)
    arr = np.empty((12, samples))
    env.reset()
    observation, _, _, _, _ = env.step(env.action_space.sample())
    j = 0
    for i in range(samples):
        j = j+1
        if j == 50:
            j = 0
            env.reset()
            observation, _, _, _, _ = env.step(env.action_space.sample())
        arr[:5, i] = observation
        action = behavior_action(observation, rng)
        arr[5:7, i] = action
        observation, reward, terminated, truncated, _ = env.step(action)
        arr[7:12, i] = observation
    X = arr[:, :samples-1]
    Y = arr[:, 1:]
    X = normalize_sas(torch.from_numpy(X).float().T).T
    Y = normalize_sas(torch.from_numpy(Y).float().T).T
    return X, Y

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

            z_dec = torch.cat([x_encoded, y_true, y_pred])
            tgt   = torch.cat([x[:, 7:12], y[:, 7:12], y[:, 7:12]])
            dec_loss = nn.MSELoss()(jepa.decoder(z_dec), tgt)
            (loss + 10.0 * dec_loss).backward()


            optimizer.step()
            optimizer.zero_grad()
            loss_sum += loss.item()
        print(epoch, ": ", loss_sum)



def cross_entropy_method(jepa, s1, a1, s2, goal, horizon, n_samples, n_elite, n_iters):
    mean = torch.full((horizon, 2), 256, device=device)
    std = torch.full((horizon, 2), 100.0, device=device)

    init = np.concatenate([s1, a1, s2])
    init_t = torch.from_numpy(init).float().to(device)
    init_t = normalize_sas(init_t)
    z_start = jepa.encoder(init_t.unsqueeze(0)).repeat(n_samples, 1)

    goal_n = normalize_state(torch.tensor([0., 0., goal[0], goal[1], goal[2]], dtype=torch.float32, device=device))[2:5]

    for i in range(n_iters):
        actions = std * torch.randn(n_samples, horizon, 2, device=device) + mean
        actions = actions.clamp(0, 512)
        z = z_start.clone()

        with torch.no_grad():
            for t in range(horizon):
                a_t = normalize_action(actions[:, t, :])
                z = jepa.predictor(torch.cat([z, a_t], dim=1))
            costs = ((jepa.decoder(z)[:, 2:5] - goal_n) ** 2).sum(dim=1)

        elites_idx = costs.topk(n_elite, largest=False).indices
        elite_actions = actions[elites_idx]
        mean = 0.7 * mean + 0.3 * elite_actions.mean(dim=0)
        std  = (0.7 * std + 0.3 * elite_actions.std(dim=0)).clamp(min=10.0)
    return mean[0]




jepa = Jepa(24, 100)
train_jepa(jepa, 100000, 1000, 100, 1e-3)

env = gym.make("gym_pusht/PushT-v0", render_mode="human")

s1, _ = env.reset()
a1 = env.action_space.sample()
s2, _, _, _, _ = env.step(a1)
goal_state = env.unwrapped.goal_pose


for i in range(200):
    mean = cross_entropy_method(jepa, s1, a1, s2, goal_state, 50, 200, 20, 50)
    s1 = s2
    a1 = mean.cpu().numpy()
    s2, _, _, _, _ = env.step(a1)
    env.render()