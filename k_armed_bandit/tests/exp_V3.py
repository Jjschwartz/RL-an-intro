"""
Experiment with the k-armed bandit with nonstationary rewards using
epsilon greedy agents running naive sample-average, incrementally computed
sample-average methods and constant step-size methods for value-estimation.
"""
from k_armed_bandit.env import KArmedBandit
from k_armed_bandit.agents import EpsilonGreedyAgent
import experiments.util as util


# Experiment parameters
K = 10
TIMESTEPS = 10000
EPISODES = 2000


def main():
    environment = KArmedBandit(K, TIMESTEPS, nonstationary=True)
    egreedy1 = EpsilonGreedyAgent(environment, TIMESTEPS)
    egreedy2 = EpsilonGreedyAgent(environment, TIMESTEPS,
                                  method="inc-sample-average")
    egreedy3 = EpsilonGreedyAgent(environment, TIMESTEPS,
                                  method="constant-step")

    rewards = {}
    opt_actions = {}

    # greedy agent
    avg_rewards, opt_a_perc = util.run_experiment(
        egreedy1, TIMESTEPS, EPISODES)
    rewards["naive"] = avg_rewards
    opt_actions["naive"] = opt_a_perc

    # epsilon-greedy with e = 0.1
    avg_rewards, opt_a_perc = util.run_experiment(
        egreedy2, TIMESTEPS, EPISODES)
    rewards["incremental"] = avg_rewards
    opt_actions["incremental"] = opt_a_perc

    # epsilon-greedy with e = 0.01
    avg_rewards, opt_a_perc = util.run_experiment(
        egreedy3, TIMESTEPS, EPISODES)
    rewards["constant"] = avg_rewards
    opt_actions["constant"] = opt_a_perc

    util.plot_experiment_results(rewards, opt_actions, TIMESTEPS)


if __name__ == "__main__":
    main()
