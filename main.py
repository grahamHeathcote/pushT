import gymnasium as gym
import gym_pusht
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from jepa import Jepa
from lightly.loss import VICRegLoss


def add_velocity(env, obs):
    agent_vel = np.array(env.unwrapped.agent.velocity)
    block_vel = np.array(env.unwrapped.block.velocity)
    return np.concatenate([obs, agent_vel, block_vel])


STATE_MEAN = torch.tensor([256., 256., 256., 256., np.pi, 0., 0., 0., 0.])
STATE_SCALE = torch.tensor([256., 256., 256., 256., np.pi, 50., 50., 50., 50.])
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

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

def behavior_action(obs, rng):
    return np.clip(obs[2:4] + rng.normal(0, 100, 2), 0, 512)

def generate_training_data(samples, seed=0):
    env = gym.make("gym_pusht/PushT-v0")
    rng = np.random.default_rng(seed)

    states = np.empty((samples, 9))
    actions = np.empty((samples, 2))
    next_states = np.empty((samples, 9))

    env.reset()
    obs, _, _, _, _ = env.step(env.action_space.sample())
    observation = add_velocity(env, obs)
    j = 0
    for i in range(samples):
        j = j + 1
        if j == 50:
            j = 0
            env.reset()
            obs, _, _, _, _ = env.step(env.action_space.sample())
            observation = add_velocity(env, obs)

        states[i] = observation
        action = behavior_action(observation, rng)
        actions[i] = action
        obs, reward, terminated, truncated, _ = env.step(action)
        observation = add_velocity(env, obs)
        next_states[i] = observation

    S = normalize_state(torch.from_numpy(states).float())
    A = normalize_action(torch.from_numpy(actions).float())
    S_next = normalize_state(torch.from_numpy(next_states).float())
    return S, A, S_next


def train_jepa(jepa, sample_size, batch_size, epochs, lr=1e-3):
    S, A, S_next = generate_training_data(sample_size)
    train_ds = TensorDataset(S, A, S_next)
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
        for s, a, s_next in train_data_loader:
            s, a, s_next = s.to(device), a.to(device), s_next.to(device)

            z = jepa.encoder(s)
            z_pred = jepa.predictor(torch.cat([z, a], dim=1))
            z_next = jepa.encoder(s_next)

            loss = loss_func(z_next, z_pred)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            loss_sum += loss.item()
        print(epoch, ": ", loss_sum)


def cross_entropy_method(jepa, s1, goal, horizon, n_samples, n_elite, n_iters):
    mean = torch.full((horizon, 2), 256, device=device)
    std = torch.full((horizon, 2), 100.0, device=device)

    s1_n = normalize_state(torch.from_numpy(s1).float().to(device))
    z_start = jepa.encoder(s1_n.unsqueeze(0)).repeat(n_samples, 1)

    goal_state = np.array([0., 0., goal[0], goal[1], goal[2], 0., 0., 0., 0.])
    goal_n = normalize_state(torch.from_numpy(goal_state).float().to(device))
    z_goal = jepa.encoder(goal_n.unsqueeze(0))

    for i in range(n_iters):
        actions = std * torch.randn(n_samples, horizon, 2, device=device) + mean
        actions = actions.clamp(0, 512)
        z = z_start.clone()

        with torch.no_grad():
            for t in range(horizon):
                a_t = normalize_action(actions[:, t, :])
                z = jepa.predictor(torch.cat([z, a_t], dim=1))
            costs = ((z - z_goal) ** 2).sum(dim=1)

        elites_idx = costs.topk(n_elite, largest=False).indices
        elite_actions = actions[elites_idx]
        mean = 0.7 * mean + 0.3 * elite_actions.mean(dim=0)
        std = (0.7 * std + 0.3 * elite_actions.std(dim=0)).clamp(min=5)
    return mean[0]


jepa = Jepa(9, 100)
train_jepa(jepa, 100000, 1000, 150, 1e-3)

env = gym.make("gym_pusht/PushT-v0", render_mode="human")

obs, _ = env.reset()
state = add_velocity(env, obs)
goal_state = env.unwrapped.goal_pose

for i in range(200):
    action = cross_entropy_method(jepa, state, goal_state, 100, 100, 20, 150)
    obs, _, _, _, _ = env.step(action.cpu().numpy())
    state = add_velocity(env, obs)
    env.render()