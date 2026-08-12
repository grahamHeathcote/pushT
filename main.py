import gymnasium as gym
import gym_pusht
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from neural_network import Network

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
    return torch.from_numpy(X).float(), torch.from_numpy(Y).float


X_train, Y_train = generate_training_data(100)

print(X_train.dtype)
encoder = Network(12, 10, 10, 8)
out = encoder(X_train[:, 20])
print(out)

train_ds = TensorDataset(X_train,Y_train)
train_data_loader = DataLoader(train_ds, batch_size=500)