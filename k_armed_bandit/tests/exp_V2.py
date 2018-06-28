"""
Experiments comparing the naive and incrementally computed sample-average
methods for action-value estimates in the k-armed bandit environment.

Specifically comparing memory and time efficiency
"""
from k_armed_bandit.env import KArmedBandit
from k_armed_bandit.agents import GreedyAgent
import experiments.util as util

# Experiment parameters
K = 10
TIMESTEPS = 10000
EPISODES = 200


def main():
    environment = KArmedBandit(K, TIMESTEPS)
    # Naive sample-average method greedy agent
    greedy_agentV1 = GreedyAgent(environment, TIMESTEPS)
    # Incrementally computed sample-average method greedy agent
    greedy_agentV2 = GreedyAgent(environment, TIMESTEPS,
                                 method="inc-sample-average")

    agents = [greedy_agentV1, greedy_agentV2]
    util.compare_agent_times(agents, TIMESTEPS, EPISODES)


if __name__ == "__main__":
    main()
