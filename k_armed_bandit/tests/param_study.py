"""
A parameter study of the different agents for the nonstationary K-armed
bandit environment.

Based off of figure 2.6 in chapter 2.10 of Suttons book

Agents and hyperparamters and value range tested:
- Greedy
    - Set Configuration:
        - Contant-step size
        - alpha = 0.1
    - Hyperparameter:
        - Optimistic initialization : [1/4, 4.0]
- Epsilon-greedy
    - Set Configuration:
        - Contant-step size
        - alpha = 0.1
    - Hyperparameter:
        - Epsilon : [1/128, 1/4]
- Upper-confidence-bound
    - Hyperparameter:
        - C = [1/16, 4]
- Gradient Bandit
    - Hyperparameter:
        - alpha = [1/32, 4]
"""
from k_armed_bandit.env import KArmedBandit
from k_armed_bandit.agents import GreedyAgent
from k_armed_bandit.agents import EpsilonGreedyAgent
from k_armed_bandit.agents import UCBAgent
from k_armed_bandit.agents import GradientBanditAgent
import matplotlib.pyplot as plt
import numpy as np


def init_values(low, high):
    vals = [low]
    while vals[-1] < high:
        vals.append(vals[-1] * 2)
    return vals


# Experiment parameters
K = 10
TIMESTEPS = 1000
EPISODES = 50
NONSTATIONARY = False
Q_INIT = init_values(1.0 / 4.0, 4.0)
EPSILONS = init_values(1.0 / 128.0, 1.0 / 4.0)
C = init_values(1.0 / 16.0, 4.0)
ALPHAS = init_values(1.0 / 32.0, 4.0)
NAMES = ["greedy", "e-greedy", "UCB", "gradient bandit"]
HPARAMS = [Q_INIT, EPSILONS, C, ALPHAS]


def get_agent(agent_name, value, env):

    if agent_name == NAMES[0]:
        return GreedyAgent(env, TIMESTEPS, q_init=value,
                           method="constant-step")
    elif agent_name == NAMES[1]:
        return EpsilonGreedyAgent(env, TIMESTEPS, epsilon=value,
                                  method="constant-step")
    elif agent_name == NAMES[2]:
        return UCBAgent(env, TIMESTEPS, c=value, method="constant-step")
    elif agent_name == NAMES[3]:
        return GradientBanditAgent(env, TIMESTEPS, alpha=value)
    else:
        raise ValueError


def test_agent(agent_name, values, env):

    print("[*] Testing {0} agent".format(agent_name))
    avg_rewards = []

    for v in values:
        agent = get_agent(agent_name, v, env)
        episode_avg = []
        for e in range(EPISODES):
            rewards, _ = agent.run_episode()
            # only average over last half of training timesteps
            # episode_avg.append(np.average(rewards[int(TIMESTEPS/2):]))
            episode_avg.append(np.average(rewards))
        avg_rewards.append(np.average(episode_avg))
        print(">>> Value = {0}, avg reward = {1}".format(v, avg_rewards[-1]))
    return avg_rewards


def plot_results(results):

    for name in results.keys():
        # averaged rewards
        y = results[name][0]
        # param values
        x = results[name][1]
        plt.plot(x, y, label=name)

    plt.ylabel("Average Reward")
    plt.xlabel("Hyperparameter Value")
    plt.xscale("log", basex=2)
    plt.legend()
    plt.show()


def main():
    env = KArmedBandit(K, TIMESTEPS, nonstationary=NONSTATIONARY)

    results = {}

    for i in range(len(NAMES)):
        res = test_agent(NAMES[i], HPARAMS[i], env)
        results[NAMES[i]] = (res, HPARAMS[i])

    plot_results(results)


if __name__ == "__main__":
    main()
