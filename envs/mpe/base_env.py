import copy

import numpy as np
import time
import gym
from collections import deque


def make_env(config: dict, benchmark=False):
    """
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Args:
        config: environment configurations.
        benchmark: whether you want to produce benchmarking data (usually only done during evaluation).

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    """
    from envs.mpe.multiagent.environment import MultiAgentEnv as MPE
    from envs.mpe.multiagent.scenarios import load

    # load scenario from script
    scenario = load(config["scenario_name"] + ".py").Scenario()
    # create world
    world = scenario.make_world(config)
    # create multiagent environment
    if benchmark:
        env = MPE(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MPE(world=world,
                  make_callback=scenario.make_world,
                  reset_callback=scenario.reset_world,
                  reward_callback=scenario.reward,
                  observation_callback=scenario.observation,
                  info_callback=scenario.get_info,
                  share_callback=scenario.share_reward,
                  landmark_cover_callback=scenario.landmark_cover_state,
                  state_callback=scenario.get_state)
    return env


class BaseMPE(gym.Env):
    """Wraps OpenAI Multi-Agent Particle env to be compatible with RLlib multi-agent."""

    def __init__(self, env_config: dict):
        """Create a new Multi-Agent Particle env compatible with RLlib.

        Args:
            env_config (dict): Arguments to pass to the underlying
                make_env.make_env instance.
        """
        self.env_config = copy.deepcopy(env_config)
        self._env = make_env(self.env_config)
        self.seed(self.env_config["seed"])
        self.num_agents = self._env.n

        self.observation_space = self._env.observation_space[0]
        self.action_space = self._env.action_space[0]

        self.eval = self.env_config["evaluation"]
        self.episode_length = self.env_config["episode_length"]
        self.ts = 0
        self.cover_rate_history = deque([0] * 5, maxlen=5)

    def reset(self, config=None):
        """Resets the env and returns observations from ready agents.

        Returns:
            obs_dict: New observations for each ready agent.
        """

        return self._env.reset(config=self.env_config if config is None else config)

    def step(self, actions):
        """Returns observations from ready agents.
        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.

        Returns:
            obs_dict:
                New observations for each ready agent.
            rew_dict:
                Reward values for each ready agent.
            done_dict:
                Done values for each ready agent.
                The special key "__all__" (required) is used to indicate env termination.
            info_dict:
                Optional info values for each agent id.
        """
        self.ts += 1
        if isinstance(self.action_space, gym.spaces.Discrete):
            action_encoding = np.zeros((self.num_agents, self.action_space.n))
            for i in range(self.num_agents):
                action_encoding[i, actions[i]] = 1
        elif isinstance(self.action_space, gym.spaces.Tuple) and \
                isinstance(self.action_space[0], gym.spaces.Discrete):
            action_encoding = np.zeros((self.num_agents, self.action_space[0].n))
            for i in range(self.num_agents):
                action_encoding[i, actions[i]] = 1
        else:
            action_encoding = actions

        obs_list, rew_list, _, info_list = self._env.step(action_encoding)

        self.cover_rate_history.appendleft(info_list[0]["cover_rate"])
        info_list = [{} for _ in range(self.num_agents)]
        if self.ts == self.episode_length:
            self.ts = 0
            done = True
            info_list[0]["cover_rate"] = np.mean(self.cover_rate_history)
            info_list[0]["num_agents"] = self.num_agents
        else:
            done = False

        return obs_list, rew_list, done, info_list

    def seed(self, seed=None):
        self._env.seed(seed)

    def render(self, mode='human'):
        time.sleep(0.05)
        self._env.render(mode=mode, close=False)
