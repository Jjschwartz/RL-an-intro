"""
This module contains agent implementations for the k armed bandit problem.

Agents implemented:
1. Greedy
    1. sample-average method
    2. incrementally computed sample-average method
    3. constant step-size
2. Epsilon-Greedy
    1. sample-average
    2. incrementally computed sample-average
    3. constant step-size
3. Upper-confidence-bound (UCB)
    1. sample-average
    2. incrementally computed sample-average
    3. constant step-size
4. Gradient Bandit Algorithm agent
"""
import numpy as np
from agents.agent import Agent

SAMPLE_AVG = "sample-average"
INC_SAMPLE_AVG = "inc-sample-average"
CONSTANT_STEP = "constant-step"


class GreedyAgent(Agent):
    """
    Greed Agent for the K-armed bandit environment
    """

    allowed_methods = set((SAMPLE_AVG, INC_SAMPLE_AVG, CONSTANT_STEP))

    def __init__(self, environment, timesteps=1000, alpha=0.1,
                 q_init=0.0, method="sample-average"):
        """
        Initialize a greedy agent

        Arguments:
        ----------
        KArmedBanditEnv environment : the environment to solve
        int timesteps : number of timesteps to run each episode for
        float alpha : step-size for action-value estimate updates
        float q_init : initial action-value estimates
        """
        self.env = environment
        self.timesteps = timesteps
        self.alpha = alpha
        self.q_init = q_init
        if method not in self.allowed_methods:
            raise ValueError("Method must be one of: {0}"
                             .format(self.allowed_methods))
        else:
            self.method = method

    def run_episode(self):
        self.env.reset()
        action_space = self.env.action_space

        rewards = np.zeros(self.timesteps)
        opt_actions = np.zeros(self.timesteps)

        q_table = np.full(len(action_space), self.q_init)
        a_n = np.zeros(len(action_space))
        r_sums = np.zeros(len(action_space))

        for t in range(self.timesteps):
            a = self.choose_action(q_table, t, a_n)
            o, r, done, info = self.env.step(a)
            rewards[t] = r
            opt_actions[t] = int(a == info["optimal_action"])
            r_sums[a] += r
            a_n[a] += 1
            q_table[a] = self.update_value(a_n[a], r_sums[a], r, q_table[a])

        return rewards, opt_actions

    def choose_action(self, q_table, t, a_n):
        """
        Choose an action from based on the action-value estimates

        N.B. t and a_n arguments are only used for UCB agent

        Arguments:
        ----------
        ndarray q_table : array of action value estimates in order of action
        int t : current timestep (used for UCB agent)
        ndarray a_n : array of times each action has been chosen

        Returns:
        --------
        int a : chosen action number
        """
        max_actions = np.argwhere(q_table == np.amax(q_table)).flatten()
        return np.random.choice(max_actions)

    def update_value(self, n, r_sum, r, q_value):
        """
        Calculate the action-value estimate for a given action

        Arguments:
        ----------
        int n : number of times action has been used
        float r_sum : sum of rewards for given action`
        float r : most recent reward recieved
        float q_value : current action-value estimate

        Returns:
        --------
        float new_q_value : updated action-value estimate
        """
        if self.method == SAMPLE_AVG:
            return r_sum / n
        elif self.method == INC_SAMPLE_AVG:
            return q_value + (1 / n) * (r - q_value)
        else:
            # Constant step-size
            return q_value + self.alpha * (r - q_value)


class EpsilonGreedyAgent(GreedyAgent):
    """
    An Epsilon-Greedy agent for the K-armed bandit environment
    """

    def __init__(self, environment, timesteps=1000, alpha=0.1,
                 q_init=0.0, method="sample-average", epsilon=0.1):
        """
        Initialize a epsilon-greedy agent

        Arguments:
        ----------
        float epsilon : the exploration probability
        """
        self.epsilon = epsilon
        super().__init__(environment, timesteps, alpha, q_init, method)

    def choose_action(self, q_table, t, a_n):
        if np.random.uniform() < self.epsilon:
            return np.random.choice(np.arange(len(q_table)))
        else:
            max_actions = np.argwhere(q_table == np.amax(q_table)).flatten()
            return np.random.choice(max_actions)


class UCBAgent(GreedyAgent):
    """
    Upper Confidence Bound Agent for the K-armed bandit environment
    """

    def __init__(self, environment, timesteps=1000, alpha=0.1,
                 q_init=0.0, method="sample-average", c=2.0):
        """
        Initialize a UCB agent

        Arguments:
        ----------
        float c : the confidence level parameter
        """
        assert c > 0
        self.c = c
        super().__init__(environment, timesteps, alpha, q_init, method)

    def choose_action(self, q_table, t, a_n):
        # check for any actions that have not been performed yet, a_n[a] = 0
        if np.any(a_n == 0) or t == 0:
            # select randomly amongst actions that have not been performed
            max_a = np.argwhere(a_n == 0).flatten()
        else:
            ucb_values = q_table + self.c * np.sqrt(np.log(t) / a_n)
            max_a = np.argwhere(ucb_values == np.amax(ucb_values)).flatten()
        return np.random.choice(max_a)


class GradientBanditAgent(Agent):
    """
    An agent for the K-armed bandit problem that uses the gradient bandit
    algorithm for action selection.
    """

    def __init__(self, environment, timesteps=1000, alpha=0.1, baseline=True):
        """
        Initialize new agent for given environment.

        Arguments:
        ----------
        KArmedBanditEnv environment : the environment to solve
        int timesteps : number of timesteps to run each problem for
        float alpha : step-size for action preference updates
        bool baseline : whether to use a reward baseline or not
        """
        assert 0 < alpha
        self.env = environment
        self.timesteps = timesteps
        self.alpha = alpha
        self.baseline = baseline

    def run_episode(self):
        self.env.reset()
        action_space = self.env.action_space

        rewards = np.zeros(self.timesteps)
        optimal_actions = np.zeros(self.timesteps)

        h_table = np.zeros(len(action_space))
        action_probs = []
        r_bar = 0

        for t in range(self.timesteps):
            action_probs = self.softmax(h_table)
            a = self.choose_action(action_probs)
            o, r, done, info = self.env.step(a)
            rewards[t] = r
            optimal_actions[t] = int(a == info["optimal_action"])
            if self.baseline:
                r_bar = self.update_baseline(t, r_bar, r)
            h_table = self.update_prefs(h_table, action_probs, a, r, r_bar)

        return rewards, optimal_actions

    def softmax(self, h_table):
        """
        Calculate the action probabilities using a softmax distribution.

        Arguments:
        ----------
        ndarray h_table : array of action preference values

        Returns:
        --------
        ndarray pi : array of action probabilities
        """
        exponents = np.exp(h_table)
        denominator = np.sum(exponents)
        pi = exponents / denominator
        return pi

    def choose_action(self, action_probs):
        """
        Choose an action based on action probabilities

        Arguments:
        ----------
        ndarray action_probs : array of action probabilities

        Returns:
        --------
        int a : chosen action number
        """
        # TODO check for more "proper" implementation
        r = np.random.sample()
        return np.argwhere(r < np.cumsum(action_probs).flatten())[0]

    def update_baseline(self, t, r_bar, r):
        """
        Returns the updated reward baseline

        Arguments:
        ----------
        int t : current timesteps
        float r_bar : current reward baseline
        float r : current reward

        Returns:
        --------
        float new_r_bar : updated reward baseline for timestep
        """
        if t == 0:
            return r_bar + r
        return r_bar + (1 / t) * (r - r_bar)

    def update_prefs(self, h_table, a_probs, a, r, r_bar):
        """
        Update the preferences using gradient bandit algorithm update.

        Arguments:
        ----------
        ndarray h_table : array of action preference values
        ndarray a_probs : array of action probabilities
        int a : the action chosen for current timestep
        float r : the reward chosen for current timestep

        Returns:
        --------
        ndarray new_h : updated array of action preference values
        """
        new_h = np.zeros(len(a_probs))
        new_h[a] = h_table[a] + self.alpha * (r - r_bar) * (1 - a_probs[a])
        for i in range(len(a_probs)):
            if i != a:
                new_h[i] = h_table[i] - self.alpha * (r - r_bar) * a_probs[i]
        return new_h
