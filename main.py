import gymnasium as gym
import gym_pusht
import numpy as np

def generate_training_data(samples):
    env = gym.make("gym_pusht/PushT-v0")
    arr = np.empty((12, samples))
    env.reset()
    for i in range(samples):
        observation, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        arr[:5, i] = observation
        action = env.action_space.sample()
        arr[5:7, i] = action
        observation, reward, terminated, truncated, _ = env.step(action)
        arr[7:12, i] = observation
    return arr

