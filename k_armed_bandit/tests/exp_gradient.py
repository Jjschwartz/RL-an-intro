"""
Experiments using the gradient bandit algorithm agent with and without
reward baselines for a couple of different step-size values.
"""
from k_armed_bandit.env import KArmedBandit
from k_armed_bandit.agents import GradientBanditAgent
import experiments.util as util

# Experiment parameters
K = 10
TIMESTEPS = 1000
EPISODES = 200
ALPHAS = [0.1, 0.4]
BASELINE = [True, False]
# shift mean value of env up so difference between baseline and no
# baseline can be seen
ENV_MEAN = 4.0


def main():
    env = KArmedBandit(K, TIMESTEPS, ENV_MEAN)
    agents = {}
    for alpha in ALPHAS:
        for b in BASELINE:
            agent = GradientBanditAgent(env, TIMESTEPS, alpha, b)
            name = "a = " + str(alpha) + ", baseline = " + str(b)
            agents[name] = agent

    rewards = {}
    opt_actions = {}

    for agent in agents.keys():
        results = util.run_experiment(agents[agent], TIMESTEPS, EPISODES)
        rewards[agent] = results[0]
        opt_actions[agent] = results[1]

    util.plot_experiment_results(rewards, opt_actions, TIMESTEPS)


if __name__ == "__main__":
    main()
