"""
Experiments using sample-average greedy and epsilon-greedy agents on the
k-armed bandit environment.
"""
from env.k_armed_bandit import KArmedBandit
from agents.k_armed_bandit_agents import GreedyAgent
from agents.k_armed_bandit_agents import EpsilonGreedyAgent
import matplotlib.pyplot as plt
import numpy as np

# Experiment parameters
K = 10
TIMESTEPS = 1000
NUM_PROBLEMS = 200
EPSILONS = [0.1, 0.01]


def plot_stuff(rewards, optimal_actions):

    tsteps = np.arange(TIMESTEPS)

    plt.subplot(2, 1, 1)
    for k in rewards.keys():
        plt.plot(tsteps, rewards[k], label=k)
    plt.ylabel("Average reward")
    plt.legend()

    plt.subplot(2, 1, 2)
    for k in optimal_actions.keys():
        plt.plot(tsteps, optimal_actions[k], label=k)
    plt.ylabel("Optimal Action proportion")
    plt.xlabel("Steps")
    plt.legend()


def main():
    environment = KArmedBandit(K, TIMESTEPS)
    greedy_agent = GreedyAgent(environment, TIMESTEPS, NUM_PROBLEMS)
    egreedy_agent0 = EpsilonGreedyAgent(environment, TIMESTEPS, NUM_PROBLEMS,
                                        EPSILONS[0])
    egreedy_agent1 = EpsilonGreedyAgent(environment, TIMESTEPS, NUM_PROBLEMS,
                                        EPSILONS[1])

    rewards = {}
    optimal_actions = {}

    average_rewards, optimal_actions_perc = greedy_agent.run_experiment()
    rewards["greedy"] = average_rewards
    optimal_actions["greedy"] = optimal_actions_perc

    average_rewards, optimal_actions_perc = egreedy_agent0.run_experiment()
    rewards["e = 0.1"] = average_rewards
    optimal_actions["e = 0.1"] = optimal_actions_perc

    average_rewards, optimal_actions_perc = egreedy_agent1.run_experiment()
    rewards["e = 0.01"] = average_rewards
    optimal_actions["e = 0.01"] = optimal_actions_perc

    plot_stuff(rewards, optimal_actions)
    plt.show()


if __name__ == "__main__":
    main()
