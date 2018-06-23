"""
Experiments comparing the naive and incrementally computed sample-average
methods for action-value estimates in the k-armed bandit environment.

Specifically comparing memory and time efficiency
"""
from env.k_armed_bandit import KArmedBandit
from agents.k_armed_bandit_agents import GreedyAgentV1
from agents.k_armed_bandit_agents import GreedyAgentV2
import util

# Experiment parameters
K = 10
TIMESTEPS = 1000
EPISODES = 2000


def main():
    environment = KArmedBandit(K, TIMESTEPS)
    # Naive sample-average method greedy agent
    greedy_agentV1 = GreedyAgentV1(environment, TIMESTEPS)
    # Incrementally computed sample-average method greedy agent
    greedy_agentV2 = GreedyAgentV2(environment, TIMESTEPS)

    agents = [greedy_agentV1, greedy_agentV2]
    util.compare_agent_efficiency(agents, TIMESTEPS, EPISODES)


if __name__ == "__main__":
    main()
