"""
This module contains the k-armed bandit environment
"""
import numpy as np


class KArmedBanditEnv(object):

    def __init__(self, k=10):
        """
        Initialize the environment.

        Arguments:
        ----------
        int k : number of actions in problems
        """
        self.k = k

    def get_action_space(self):
        """
        Return the action space for the environment. Actions are defined as
        ints from 0 to k.

        Returns:
        --------
        list actions : list of possible actions
        """
        return list(range(self.k))

    def get_next_problem(self):
        """
        Generate and return the next problem

        Returns:
        --------
        KArmedBanditProblem problem : a newly generated problem
        """
        return KArmedBanditProblem(self.k)


class KArmedBanditProblem(object):

    def __init__(self, k):
        """
        Initialize a k armed bandit problem.

        Arguments:
        ----------
        int k : number of actions in problems
        """
        self.k = k
        self.action_values = self._get_action_values(k)

    def _get_action_values(self, k):
        """
        Initialize the action values from random samples of Gaussian with mean
        of 0 and variance of 1.

        Arguments:
        ----------
        int k : number of actions in problems

        Returns:
        --------
        list action_values : list of action_values in order of action number
        """
        return np.random.normal(loc=0.0, scale=1.0, size=k)

    def _get_action_reward(self, a):
        """
        Get the reward for a given action.

        Arguments:
        ----------
        int a : the action number

        Returns:
        --------
        float reward : the reward for performing action a
        """
        return np.random.normal(loc=self.action_values[a], scale=1.0)

    def perform_action(self, a):
        """
        Perform action a on environment problem and get reward.

        Arguments:
        ----------
        int a : the action number

        Returns:
        --------
        float reward : the reward for performing action a
        """
        assert a >= 0 and a < self.k
        return self._get_action_reward(a)

    def get_optimal_action(self):
        """
        Returns the optimal action number for agent evaluation purposes.

        Returns:
        --------
        int a : action number corresponding to the action with highest value
        """
        return np.argmax(self.action_values)
