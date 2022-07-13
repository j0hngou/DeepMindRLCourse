import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class MultiArmedBandit:
    """
    MultiArmedBandit class.
    """
    def __init__(self, k, means=None, stds=None):
        """
        Initializes the MultiArmedBandit.
        params:
            k: number of arms.
            means: list of means for each arm.
            stds: list of stds for each arm.
        """
        self.k = k
        self.arms = np.zeros(k)
        self.times = np.zeros(k)
        if means is None:
            self.means = (np.random.uniform(-1, 1, k))
        else:
            assert(len(means) == k), "Means must be of length k."
            self.means = means
        if stds is None:
            self.stds = np.ones(k)
        else:
            assert(len(stds) == k), "Stds must be of length k."
            self.stds = stds
    
    def pull(self, arm) -> float:
        """
        Pulls the arm and returns the reward.
        params:
            arm: the arm to pull.
            return: the reward.
        """
        bandit = np.random.normal(self.means, self.stds)[arm]
        return bandit
    
    def increment(self, arm) -> None:
        """
        Increments the arm's times.
        params:
            arm: the arm to increment.
            return: None.
        """
        self.times[arm] += 1

    def pull_and_increment(self, arm) -> float:
        """
        Pulls the arm and increments the arm's times.
        params:
            arm: the arm to pull.
            return: the reward.
        """
        bandit = self.pull(arm)
        self.increment(arm)
        return bandit
