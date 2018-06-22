"""
This module contains agent implementations for the k armed bandit problem.

Agents implemented:
1. Greedy using sample-average method
2. Epsilon-Greedy using sample-average method
"""
import numpy as np


class GreedyAgent(object):

    def __init__(self, environment, timesteps=1000, num_problems=2000):
        """
        Initialize a new agent with given environment.

        Arguments:
        ----------
        KArmedBanditEnv environment : the environment to solve
        int timesteps : number of timesteps to run each problem for
        int num_problems : number of different problems to run agent on
        """
        self.env = environment
        self.timesteps = timesteps
        self.num_problems = num_problems

    def run_experiment(self):
        """
        Run the experiment.

        Returns:
        --------
        ndarray average_rewards : the reward for each timestep averaged over
            all problem runs
        ndarray optimal_actions_perc : the proportion of optimal actions
            performed for each timestep averaged over all problem runs
        """
        print("[+] Running Experiment")

        average_rewards = np.zeros(self.timesteps)
        optimal_actions_perc = np.zeros(self.timesteps)

        for p in range(self.num_problems):
            rewards, optimal_actions = self.run_episode()
            average_rewards += rewards
            optimal_actions_perc += optimal_actions

            if p % 50 == 0:
                print("[+] Finished Problem {0}".format(p))

        average_rewards = average_rewards / self.num_problems
        optimal_actions_perc = optimal_actions_perc / self.num_problems

        return average_rewards, optimal_actions_perc

    def run_episode(self):
        """
        Run a full episode of k-armed bandit environment

        Returns:
        --------
        ndarray rewards : the reward for each timestep
        ndarray optimal_actions : whether or not optimal action was performed
            for each timestep
        """
        self.env.reset()
        action_space = self.env.action_space
        optimal_action = self.env.optimal_action

        rewards = np.zeros(self.timesteps)
        optimal_actions = np.zeros(self.timesteps)

        q_table = np.zeros(len(action_space))
        a_count = np.zeros(len(action_space))

        for t in range(self.timesteps):
            a = self.choose_action(q_table)
            o, r, done, _ = self.env.step(a)
            rewards[t] = r
            optimal_actions[t] = int(a == optimal_action)
            a_count[a] += 1
            q_table[a] = self.update_value(a_count[a], rewards[t], q_table[a])

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

    def update_value(self, a_count, reward, q_value):
        """
        Update the action value estimate for a given action based on gained
        reward using the sample-average method.

        Arguments:
        ----------
        int a : the action number performed
        int a_count : the number of times the action has been performed,
            including the current timestep
        float reward : the reward gained for performing action
        float q_value : current action value estimate

        Returns:
        float new_a_value : updated action value estimate
        """
        return q_value + (1 / a_count) * (reward - q_value)


class EpsilonGreedyAgent(GreedyAgent):

    def __init__(self, environment, timesteps=1000, num_problems=2000,
                 epsilon=0.1):
        """
        Initialize a new epsilon-greedy agent with given environment.

        Arguments:
        ----------
        KArmedBanditEnv environment : the environment to solve
        int timesteps : number of timesteps to run each problem for
        int num_problems : number of different problems to run agent on
        float epsilon : the exploration probability for agent
        """
        self.epsilon = epsilon
        super().__init__(environment, timesteps, num_problems)

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
