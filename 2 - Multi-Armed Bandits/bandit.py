import random
import numpy as np

class Dist():
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    # return a value sampled from the Normal distrition with this object's mean and std dev (mu and sigma)
    # this sampled value represents the reward
    def sample(self):
        return np.random.normal(loc=self.mu, scale=self.sigma)

    def __str__(self):
        return str(self.mu) + '|' + str(self.sigma)


class Bandit():
    def __init__(self, k):
        # create k distrubutions with random mean ∈ [-5, 5] and a standard deviation of 1
        self.distributions = [Dist(random.randint(-5, 5), 1) for _ in range(k)]

        # determine which probability distribution has the highest mean
        self.optimal_a = 0
        for i in range(k):
            pd = self.distributions[i]
            if pd.mu > self.distributions[self.optimal_a].mu:
                self.optimal_a = i

    # sample from the proper reward distribution when the agent selects an action
    def act(self, a):
        return self.distributions[a].sample()
