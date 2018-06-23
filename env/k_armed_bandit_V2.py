"""
From "Reinforcement Learning: An Introduction" Chapter 2.5

Design and conduct an experiment to demonstrate the difficulties
that sample-average methods have for nonstationary problems. Use a modified
version of the 10-armed testbed in which all the q ∗ (a) start out equal and
then take independent random walks (say by adding a normally distributed
increment with mean zero and standard deviation 0.01 to all the q ∗ (a) on each
step). Prepare plots like Figure 2.2 for an action-value method using sample
averages, incrementally computed, and another action-value method using a
constant step-size parameter, α = 0.1. Use ε = 0.1 and longer runs, say of
10,000 steps.
"""
import numpy as np
from env.core import Env


class KArmedBanditV2(Env):
    """
    The classic K-armed Bandit problem with a twist. The reward distribution
    changes over time (nonstationary).

    A single state environment where at each timestep the agent must choose one
    of k actions with the aim of maximizing the average reward over time. The
    rewards of each action are nondetermnistic but there is an optimal action
    which provides the highest reward over the long term.

    Actual action-values, q*(a), start out equal and then take independent
    walks.

    An episode ends after a specified number of timesteps.
    When the environment is reset, the action-values are reset are, so the
    optimal action changes between episodes.

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
        self.reset()

    def reset(self):
        self.steps_taken = 0
        initial_q_value = np.random.normal()
        self.action_values = np.full(self.k, initial_q_value)
        return self.observation_space

    def step(self, action):
        assert action >= 0 and action < self.k
        self.steps_taken += 1
        reward = np.random.normal(loc=self.action_values[action], scale=1.0)
        done = self.steps_taken >= self.timesteps
        # optimal action may change each step, due to nonstationary reward
        optimal_action = np.argmax(self.action_values)
        # update the reward distribution of each action
        self.action_values += self._get_reward_increment()
        return (self.observation_space, reward, done,
                {"optimal_action": optimal_action})

    def _get_reward_increment(self):
        """
        Generates the randomly walk increment for the reward disctribution
        from randomly sampling a normal distribution mean of 0.0 and standard
        deviation 0.01.
        """
        increment = np.random.normal(loc=0.0, scale=0.01, size=self.k)
        return increment
