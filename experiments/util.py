import numpy as np
import matplotlib.pyplot as plt
import time


def run_experiment(agent, timesteps, num_episodes, verbose=True):
    """
    Run an experiment using agent, for given number of episodes.

    Arguments:
    ----------
    Agent agent : the agent to run experiment with
    int timesteps : number of timesteps each episode is run for
    int num_episodes : the number of episodes to run experiment for
    boolean verbose : whether to print progress messages of not

    Returns:
    --------
    ndarray average_rewards : the reward for each timestep averaged over
        all problem runs
    ndarray optimal_actions_perc : the proportion of optimal actions
        performed for each timestep averaged over all problem runs
    """
    if verbose:
        print("[+] Running Experiment")

    average_rewards = np.zeros(timesteps)
    optimal_actions_perc = np.zeros(timesteps)

    for p in range(num_episodes):
        rewards, optimal_actions = agent.run_episode()
        average_rewards += rewards
        optimal_actions_perc += optimal_actions

        if verbose and p % 50 == 0:
            print("[+] Finished Problem {0}".format(p))

    average_rewards = average_rewards / num_episodes
    optimal_actions_perc = optimal_actions_perc / num_episodes

    return average_rewards, optimal_actions_perc


def plot_experiment_results(rewards, optimal_actions, timesteps):
    """
    Sets up and displays plots for average rewards and optimal action percent
    per timestep.

    Arguments:
    ----------
    ndarray rewards : the reward for each timestep averaged over all episodes
    ndarray optimal_actions_perc : the proportion of optimal actions
        performed for each timestep averaged over all episodes
    int timesteps : number of timesteps each episode is run for
    """

    tsteps = np.arange(timesteps)

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
    plt.show()


def compare_agent_times(agents, timesteps, episodes):
    """
    Displays the time taken to run a list of agents.
    """

    run_times = np.zeros(len(agents))

    for i in range(len(agents)):
        print("=> Testing agent {0}".format(i))
        start_t = time.time()
        run_experiment(agents[i], timesteps, episodes, verbose=False)
        run_times[i] = time.time() - start_t
        print("=> Time taken = {0}".format(run_times[i]))
