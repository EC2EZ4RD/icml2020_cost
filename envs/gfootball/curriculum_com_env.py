import gym
import numpy as np
from ray.rllib.utils.spaces.repeated import Repeated

from .curriculum_base_env import BaseFootballCurriculumEnv


class FootballCurriculumComEnv(BaseFootballCurriculumEnv):
    """Wraps Google Football env to be compatible with RLlib multi-agent communication."""

    def __init__(self, env_config):
        BaseFootballCurriculumEnv.__init__(self, env_config)
        self.max_num_agents = self.env_config["custom_configs"]["max_num_agents"]
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(19),) * self.max_num_agents)
        self.observation_space = Repeated(
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(115,)), max_len=self.max_num_agents)

    def reset(self):
        obs_list = BaseFootballCurriculumEnv.reset(self)
        return self.convert_obs(obs_list)

    def step(self, actions):
        # deal with unavailable agents
        act = list(actions)[:self.num_agents]
        obs_list, rew_list, done, info = BaseFootballCurriculumEnv.step(self, act)
        all_rewards = sum(rew_list)
        rew_list = rew_list.tolist() + [0.0] * (self.max_num_agents - self.num_agents)
        rewards = {i: rew for i, rew in enumerate(rew_list)}
        info["rewards"] = rewards
        info["num_agents"] = self.num_agents

        return self.convert_obs(obs_list), all_rewards, done, info

    def convert_obs(self, obs_list):
        return [obs_list[i] for i in range(self.num_agents)]
