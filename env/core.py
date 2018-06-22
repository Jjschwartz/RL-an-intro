"""
The core module for the environments.
Contains the abstract implementation of the environment class, to provide a
framework for future environments.

Inspired by the OpenAI gym implementation.
"""


class Env(object):
    """
    The main environment class.

    The Main environment class methods are:

        step
        reset

    The main environment class attributes are:

        action_space: The set of valid actions
        observation_space: The set of valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    """

    # optional to change
    reward_range = (-float('inf'), float('inf'))

    # must be set for ALL environments
    action_space = None
    observation_space = None

    def step(self, action):
        """
        Run one timestep of the envionment.

        Accepts an action and returns a tuple (observation, reward, done, info)

        Arguments:
        ----------
        action: a valid action in action_space

        Returns:
        --------
        observation: agent's observation of current environment
        reward: amount of reward for taken action
        done: whether the episode has ended
        info: contais auxiliary diagnostic information
        """
        raise NotImplementedError

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.

        Returns:
        --------
        observation: initial observation of the environment
        """
        raise NotImplementedError
