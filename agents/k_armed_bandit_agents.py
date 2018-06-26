"""
This module contains agent implementations for the k armed bandit problem.

Agents implemented:
1. Greedy using sample-average method
2. Greedy using incrementally computed sample-average method
3. Epsilon-Greedy using sample-average method
4. Epsilon-Greedy using incrementally computed sample-average method
5. Epsilon-Greedy using constant step-size
"""
import numpy as np
from agents.agent import Agent


class GreedyAgentV1(Agent):
    """
    Greedy agent for the K-armed bandit environment using sample-average method
    of action-value estimate

        Qn(a) = sum of rewards upto nth time a is used / n

    Where:
        n - number of times action a has been used
    """

    def __init__(self, environment, timesteps=1000):
        """
        Initialize a new agent with given environment.

        Arguments:
        ----------
        KArmedBanditEnv environment : the environment to solve
        int timesteps : number of timesteps to run each episode for
        int num_episodes : number of episodes to run agent on
        """
        self.env = environment
        self.timesteps = timesteps

    def run_episode(self):
        self.env.reset()
        action_space = self.env.action_space

        rewards = np.zeros(self.timesteps)
        optimal_actions = np.zeros(self.timesteps)
        # to store sum of reward for each action for sample-average method
        reward_sums = np.zeros(len(action_space))

        q_table = np.zeros(len(action_space))
        a_n = np.zeros(len(action_space))

        for t in range(self.timesteps):
            a = self.choose_action(q_table)
            o, r, done, info = self.env.step(a)
            rewards[t] = r
            reward_sums[a] += r
            optimal_actions[t] = int(a == info["optimal_action"])
            a_n[a] += 1
            q_table[a] = self.update_value(a_n[a], reward_sums[a])

        return rewards, optimal_actions

    def choose_action(self, q_table):
        """
        Choose an action from Q table (action value estimate table), greedily.
        In the case where two or more actions have same value, one of the tied
        actions is chosen randomly.

        Arguments:
        ----------
        ndarray q_table : array of action value estimates in order of action
            number

        Returns:
        --------
        int a : chosen action number
        """
        max_actions = np.argwhere(q_table == np.amax(q_table)).flatten()
        return np.random.choice(max_actions)

    def update_value(self, n, rewards):
        """
        Update the action value estimate for a given action based on gained
        reward using the sample-average method.

        Arguments:
        ----------
        int n : the number of times the action has been performed, including
            the current timestep
        ndarray rewards : sum of rewards gained for performing action

        Returns:
        float new_a_value : updated action value estimate
        """
        return rewards / n


class GreedyAgentV2(GreedyAgentV1):
    """
    Greedy agent for the K-armed bandit environment using incrementally
    computed sample-average method of action-value estimate

        Qn+1(a) = Qn + (1 / n) * [Rn - Qn]

    Where:
        n - number of times action a has been used
    """

    def run_episode(self):
        self.env.reset()
        action_space = self.env.action_space

        rewards = np.zeros(self.timesteps)
        optimal_actions = np.zeros(self.timesteps)

        q_table = np.zeros(len(action_space))
        a_count = np.zeros(len(action_space))

        for t in range(self.timesteps):
            a = self.choose_action(q_table)
            o, r, done, info = self.env.step(a)
            rewards[t] = r
            optimal_actions[t] = int(a == info["optimal_action"])
            a_count[a] += 1
            q_table[a] = self.update_value(a_count[a], rewards[t], q_table[a])

        return rewards, optimal_actions

    def update_value(self, n, reward, q_value):
        """
        Update the action value estimate for a given action based on gained
        reward using the incrementally computed sample-average method.

        Arguments:
        ----------
        int n : the number of times the action has been performed, including
            the current timestep
        float reward : the most recent reward gained for performing action
        float q_value : current action value estimate

        Returns:
        float new_a_value : updated action value estimate
        """
        return q_value + (1 / n) * (reward - q_value)


class EpsilonGreedyAgentV1(GreedyAgentV1):
    """
    Epsilon-Greedy agent for the K-armed bandit environment using the
    sample-average method of action-value estimate
    """

    def __init__(self, environment, timesteps=1000, epsilon=0.1):
        """
        Initialize a new epsilon-greedy agent with given environment.

        Arguments:
        ----------
        KArmedBanditEnv environment : the environment to solve
        int timesteps : number of timesteps to run each problem for
        float epsilon : the exploration probability for agent
        """
        self.epsilon = epsilon
        super().__init__(environment, timesteps)

    def choose_action(self, q_table):
        """
        Choose an action from Q table (action value estimate table), greedily,
        but also select action uniformly at random with probability = epsilon.
        In the case where two or more actions have same value, one of the tied
        actions is chosen randomly.

        Arguments:
        ----------
        ndarray q_table : array of action value estimates in order of action
            number

        Returns:
        --------
        int a : chosen action number
        """
        if np.random.uniform() < self.epsilon:
            return np.random.choice(np.arange(len(q_table)))
        else:
            max_actions = np.argwhere(q_table == np.amax(q_table)).flatten()
            return np.random.choice(max_actions)


class EpsilonGreedyAgentV2(GreedyAgentV2):
    """
    Epsilon-Greedy agent for the K-armed bandit environment using incrementally
    computed sample-average method of action-value estimate
    """

    def __init__(self, environment, timesteps=1000, epsilon=0.1):
        """
        Initialize a new epsilon-greedy agent with given environment.

        Arguments:
        ----------
        KArmedBanditEnv environment : the environment to solve
        int timesteps : number of timesteps to run each problem for
        float epsilon : the exploration probability for agent
        """
        self.epsilon = epsilon
        super().__init__(environment, timesteps)

    def choose_action(self, q_table):
        """
        Choose an action from Q table (action value estimate table), greedily,
        but also select action uniformly at random with probability = epsilon.
        In the case where two or more actions have same value, one of the tied
        actions is chosen randomly.

        Arguments:
        ----------
        ndarray q_table : array of action value estimates in order of action
            number

        Returns:
        --------
        int a : chosen action number
        """
        if np.random.uniform() < self.epsilon:
            return np.random.choice(np.arange(len(q_table)))
        else:
            max_actions = np.argwhere(q_table == np.amax(q_table)).flatten()
            return np.random.choice(max_actions)


class EpsilonGreedyAgentV3(EpsilonGreedyAgentV2):
    """
    Epsilon-Greedy agent for the K-armed bandit environment using constant
    step-size incrementally computed sample-average of action-value estimates.
    """

    def __init__(self, environment, timesteps=1000, epsilon=0.1, alpha=0.1):
        """
        Initialize a new epsilon-greedy agent with given environment.

        Arguments:
        ----------
        KArmedBanditEnv environment : the environment to solve
        int timesteps : number of timesteps to run each problem for
        float epsilon : the exploration probability for agent
        float alpha : step-size for action-value estimate updates
        """
        assert 0 < alpha and alpha <= 1
        self.alpha = alpha
        super().__init__(environment, timesteps, epsilon)

    def update_value(self, n, reward, q_value):
        """
        Update the action value estimate for a given action based on gained
        reward using the incrementally computed sample-average method with
        constant step-size
        """
        return q_value + self.alpha * (reward - q_value)


class UCBAgent(Agent):
    """
    Upper Confidence Bound Agent for the K-armed bandit environment using
    constant step-size action-value estimates.
    """

    def __init__(self, environment, timesteps=1000, alpha=0.1, c=2):
        """
        Initialize a new UCB agent with given environment.

        Arguments:
        ----------
        KArmedBanditEnv environment : the environment to solve
        int timesteps : number of timesteps to run each problem for
        float alpha : step-size for action-value estimate updates
        float c : the confidence level parameter, c > 0
        """
        assert 0 < alpha and alpha <= 1
        assert c > 0
        self.env = environment
        self.timesteps = timesteps
        self.alpha = alpha
        self.c = c

    def run_episode(self):
        self.env.reset()
        action_space = self.env.action_space

        rewards = np.zeros(self.timesteps)
        optimal_actions = np.zeros(self.timesteps)

        q_table = np.zeros(len(action_space))
        a_n = np.zeros(len(action_space))

        for t in range(self.timesteps):
            a = self.choose_action(q_table, t, a_n)
            o, r, done, info = self.env.step(a)
            rewards[t] = r
            optimal_actions[t] = int(a == info["optimal_action"])
            a_n[a] += 1
            q_table[a] = self.update_value(a_n[a], r, q_table[a])

        return rewards, optimal_actions

    def update_value(self, n, reward, q_value):
        return q_value + self.alpha * (reward - q_value)

    def choose_action(self, q_table, t, a_n):
        """
        Choose an action using UCB action selection

        At = argmax_a [Qt(a) + c * sqrt{ [ln t] / [Nt(a)]}]

        Arguments:
        ----------
        ndarray q_table : array of action value estimates in order of action
            number
        int t : current timestep
        ndarray a_n : array of times each action has been chosen

        Returns:
        --------
        int a : chosen action number
        """
        # check for any actions that have not been performed yet, a_n[a] = 0
        if np.any(a_n == 0) or t == 0:
            # select randomly amongst actions that have not been performed
            max_a = np.argwhere(a_n == 0).flatten()
            return np.random.choice(max_a)
        else:
            ucb_values = q_table + self.c * np.sqrt(np.log(t) / a_n)
            max_a = np.argwhere(ucb_values == np.amax(ucb_values)).flatten()
            return np.random.choice(max_a)
