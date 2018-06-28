from k_armed_bandit.env import KArmedBandit
from k_armed_bandit.agents import EpsilonGreedyAgent
from k_armed_bandit.agents import UCBAgent
import experiments.util as util

# Experiment parameters
K = 10
TIMESTEPS = 1000
EPISODES = 200


def main():
    environment = KArmedBandit(K, TIMESTEPS)
    egreedy = EpsilonGreedyAgent(environment, TIMESTEPS,
                                 method="constant-step")
    ucb = UCBAgent(environment, TIMESTEPS, method="constant-step")

    rewards = {}
    opt_actions = {}

    # greedy agent
    avg_rewards, opt_a_perc = util.run_experiment(egreedy, TIMESTEPS, EPISODES)
    rewards["e-greedy e = 0.1"] = avg_rewards
    opt_actions["e-greedy e = 0.1"] = opt_a_perc

    # UCB Agent
    avg_rewards, opt_a_perc = util.run_experiment(ucb, TIMESTEPS, EPISODES)
    rewards["UCB c = 2"] = avg_rewards
    opt_actions["UCB c = 2"] = opt_a_perc

    util.plot_experiment_results(rewards, opt_actions, TIMESTEPS)


if __name__ == "__main__":
    main()
