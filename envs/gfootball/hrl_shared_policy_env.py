import gym
import numpy as np

from .hrl_base_env import BaseFootballHierarchicalEnv


class FootballHierarchicalMultiAgentEnv(BaseFootballHierarchicalEnv):
    """
    High-level: parameter sharing PPO policy
    Low-level: context-based parameter sharing PPO policy
    """

    def __init__(self, env_config):
        BaseFootballHierarchicalEnv.__init__(self, env_config)

        # high-level obs and act spaces
        self.high_level_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(115,))
        if self.context_type == "continuous":
            self.high_level_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.context_size,))
        elif self.context_type == "discrete":
            self.high_level_action_space = gym.spaces.Discrete(self.context_size)
        else:
            raise NotImplementedError("Unsupported high-level action space.")

    @property
    def high_level_obs(self):
        return {f"high_level_{i}": self.env_obs[i] for i in range(self.num_agents)}

    def high_level_actions(self, action_dict):
        return [action_dict[f"high_level_{i}"] for i in range(self.num_agents)]

    @property
    def high_level_rewards(self):
        return {f"high_level_{i}": self.low_level_accumulated_rew[i] for i in range(self.num_agents)}

    @property
    def high_level_infos(self):
        return {f"high_level_{i}": self.env_info for i in range(self.num_agents)}
