"""
Experiment with the k-armed bandit with nonstationary rewards using
epsilon greedy agents running naive sample-average, incrementally computed
sample-average methods and constant step-size methods for value-estimation.
"""
from env.k_armed_bandit_V2 import KArmedBanditV2
from agents.k_armed_bandit_agents import EpsilonGreedyAgentV1
from agents.k_armed_bandit_agents import EpsilonGreedyAgentV2
from agents.k_armed_bandit_agents import EpsilonGreedyAgentV3
import util


# Experiment parameters
K = 10
TIMESTEPS = 10000
EPISODES = 1000
EPSILON = 0.1
ALPHA = 0.1


def main():
    environment = KArmedBanditV2(K, TIMESTEPS)
    egreedy1 = EpsilonGreedyAgentV1(environment, TIMESTEPS, EPSILON)
    egreedy2 = EpsilonGreedyAgentV2(environment, TIMESTEPS, EPSILON)
    egreedy3 = EpsilonGreedyAgentV3(environment, TIMESTEPS, EPSILON, ALPHA)

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
