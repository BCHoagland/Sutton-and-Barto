import random

#            A  B  C  D  E
positions = [0, 0, 0, 0, 0]

def is_terminal(ind):
    return ind == -1 or ind == len(positions)

def reward(ind):
    return 1 if ind == len(positions) else 0

def reset():
    return len(positions) // 2

def step(ind):
    new_ind = ind + 1 if random.random() < 0.5 else ind - 1
    r = reward(new_ind)
    done = True if is_terminal(new_ind) else False
    return new_ind, r, done
