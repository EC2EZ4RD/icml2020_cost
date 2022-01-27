import gym
import numpy as np
from ray.rllib.utils.spaces.repeated import Repeated

from .hrl_base_env import BaseFootballHierarchicalEnv
from .curriculum_base_env import BaseFootballCurriculumEnv


class FootballCurriculumHierarchicalComEnv(BaseFootballHierarchicalEnv, BaseFootballCurriculumEnv):

    def __init__(self, env_config):
        self.task = None
        BaseFootballHierarchicalEnv.__init__(self, env_config)
        self.max_num_agents = self.env_config["custom_configs"]["max_num_agents"]
        self.high_level_observation_space = Repeated(
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(115,)), max_len=self.max_num_agents)
        if self.context_type == "continuous":
            self.high_level_action_space = gym.spaces.Tuple(
                [gym.spaces.Box(low=-1.0, high=1.0, shape=(self.context_size,)) for _ in range(self.num_agents)])
        elif self.context_type == "discrete":
            self.high_level_action_space = gym.spaces.Tuple(
                [gym.spaces.Discrete(self.context_size) for _ in range(self.num_agents)])
        else:
            raise NotImplementedError("Unsupported high-level action space.")

    def reset(self):
        self.env_obs = BaseFootballCurriculumEnv.reset(self)
        self.env_rewards = [0.0] * self.num_agents
        self.low_level_accumulated_rew = {i: 0.0 for i in range(self.num_agents)}
        return self.high_level_obs

    @property
    def high_level_obs(self):
        return {"high_level_policy": [self.env_obs[i] for i in range(self.num_agents)]}

    def high_level_actions(self, action_dict):
        return list(action_dict["high_level_policy"])[:self.num_agents]

    @property
    def high_level_rewards(self):
        return {"high_level_policy": sum(self.low_level_accumulated_rew.values())}

    @property
    def high_level_infos(self):
        rew_list = [self.low_level_accumulated_rew[i] for i in range(self.num_agents)] + \
                   [0.0] * (self.max_num_agents - self.num_agents)
        rewards = {i: rew for i, rew in enumerate(rew_list)}
        self.env_info["rewards"] = rewards
        self.env_info["num_agents"] = self.num_agents
        return {"high_level_policy": self.env_info}
