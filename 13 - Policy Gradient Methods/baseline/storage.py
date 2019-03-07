import torch
import numpy as np

states = []
actions = []
rewards = []

def clear_history():
    del states[:]
    del actions[:]
    del rewards[:]

def store(s, a, r):
    states.append(s)
    actions.append(a)
    rewards.append(r)

def get_history():
    return torch.FloatTensor(np.array(states)), torch.FloatTensor(np.array(actions)), torch.FloatTensor(np.array(rewards))
