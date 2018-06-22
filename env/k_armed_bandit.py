from env.core import Env
import numpy as np


class KArmedBandit(Env):
    """
    The classic Reinforcement Learning problem.
    Introduce in chapter 2 of Reinforcement Learning: an Introduction.

    A single state environment where at each timestep the agent must choose one
    of k actions with the aim of maximizing the average reward over time. The
    rewards of each action are nondetermnistic but there is an optimal action
    which provides the highest reward over the long term.

    Actual Action-values, q*(a), are chosen at random from a Gaussian
    distribution with mean = 0 and variance = 1.

    The reward, R, for any given action, A, at any timestep, t, is chosen
    randomly from a Gaussian distribution with mean = q*(A) and variance = 1.

    An episode ends after a specified number of timesteps.
    When the environment is reset, the action-values are again chosen at
    random, so optimal action changes between episodes.

    This environment contains only a single state, so the agent observation is
    redundant and so is just True.
    """

    def __init__(self, k=10, timesteps=1000):
        """
        Initialize the environment.

        Arguments:
        ----------
        int k : number of actions in problems
        int timesteps : number of steps to perform per episode
        """
        self.k = k
        self.timesteps = timesteps
        self.action_space = list(range(k))
        self.observation_space = True

    def reset(self):
        self.steps_taken = 0
        self.action_values = np.random.normal(loc=0.0, scale=1.0, size=self.k)
        self.optimal_action = np.argmax(self.action_values)
        return self.observation_space

    def step(self, action):
        assert action >= 0 and action < self.k
        self.steps_taken += 1
        reward = np.random.normal(loc=self.action_values[action], scale=1.0)
        done = self.steps_taken >= self.timesteps
        return (self.observation_space, reward, done,
                {"optimal_action": self.optimal_action})
