import gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from .curriculum_base_env import BaseFootballCurriculumEnv


class FootballCurriculumMultiAgentEnv(BaseFootballCurriculumEnv, MultiAgentEnv):
    """Wraps Google Football env to be compatible with RLlib multi-agent."""

    def __init__(self, env_config):
        BaseFootballCurriculumEnv.__init__(self, env_config)
        self.action_space = gym.spaces.Discrete(19)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(115,))

    def reset(self):
        obs_list = BaseFootballCurriculumEnv.reset(self)
        return self.convert_obs(obs_list)

    def step(self, actions):
        act = [actions[f"agent_{i}"] for i in range(self.num_agents)]
        obs_list, rew_list, done, info = BaseFootballCurriculumEnv.step(self, act)
        reward = {f"agent_{i}": rew_list[i] for i in range(self.num_agents)}
        done = {"__all__": done}
        info = {f"agent_{i}": info for i in range(self.num_agents)}

        return self.convert_obs(obs_list), reward, done, info

    def convert_obs(self, obs_list):
        return {f"agent_{i}": obs_list[i] for i in range(self.num_agents)}
