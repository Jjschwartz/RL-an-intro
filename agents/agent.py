

class Agent(object):
    """
    The base agent class.

    Implemented to allow for easier evaluation of agents by providing a
    consistent interface.

    Main method:
        run_episode
    """

    def run_episode(self):
        """
        Run a full episode of the agent and it's environment

        Returns:
        --------
        ndarray rewards : the reward for each timestep
        ndarray optimal_actions : whether or not optimal action was performed
            for each timestep
        """
        raise NotImplementedError
