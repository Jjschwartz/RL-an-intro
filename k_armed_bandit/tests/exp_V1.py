"""
Experiments using sample-average greedy and epsilon-greedy agents on the
k-armed bandit environment.
"""
from k_armed_bandit.env import KArmedBandit
from k_armed_bandit.agents import GreedyAgent
from k_armed_bandit.agents import EpsilonGreedyAgent
import experiments.util as util

# Experiment parameters
K = 10
TIMESTEPS = 1000
EPISODES = 2000
EPSILONS = [0.1, 0.01]


def main():
    environment = KArmedBandit(K, TIMESTEPS)
    greedy_agent = GreedyAgent(environment, TIMESTEPS)
    egreedy_agent0 = EpsilonGreedyAgent(environment, TIMESTEPS,
                                        epsilon=EPSILONS[0])
    egreedy_agent1 = EpsilonGreedyAgent(environment, TIMESTEPS,
                                        epsilon=EPSILONS[1])

    rewards = {}
    opt_actions = {}

    # greedy agent
    avg_rewards, opt_a_perc = util.run_experiment(
        greedy_agent, TIMESTEPS, EPISODES)
    rewards["greedy"] = avg_rewards
    opt_actions["greedy"] = opt_a_perc

    # epsilon-greedy with e = 0.1
    avg_rewards, opt_a_perc = util.run_experiment(
        egreedy_agent0, TIMESTEPS, EPISODES)
    rewards["e = 0.1"] = avg_rewards
    opt_actions["e = 0.1"] = opt_a_perc

    # epsilon-greedy with e = 0.01
    avg_rewards, opt_a_perc = util.run_experiment(
        egreedy_agent1, TIMESTEPS, EPISODES)
    rewards["e = 0.01"] = avg_rewards
    opt_actions["e = 0.01"] = opt_a_perc

    util.plot_experiment_results(rewards, opt_actions, TIMESTEPS)


if __name__ == "__main__":
    main()
