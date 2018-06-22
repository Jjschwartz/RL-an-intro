# k-armed bandit

This module contains implementations of the k-armed bandit problem and also some agents for solving the problem.

## The problem definition

The problem is setup as follows:
- At each timestep the agent must choose one of k actions with the aim of maximizing the average reward over time.
- Actual Action-values, q*(a), are chosen at random from a Gaussian distribution with mean = 0 and variance = 1.
- The reward, R, for any given action, A, at any timestep, t, is chosen randomly from a Gaussian distribution with mean = q*(A) and variance = 1.

## The environment

The agent can interact with the environment by choosing an action at each timestep and then the environment will return a reward.

Environment parameters:
- The number of different randomly generated problems
- Timesteps per problem (aka run)
- The number of actions, k
