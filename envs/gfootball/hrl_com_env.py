import gym
import numpy as np

from .hrl_base_env import BaseFootballHierarchicalEnv


class FootballHierarchicalComEnv(BaseFootballHierarchicalEnv):
    """
    High-level: attention-base communication PPO policy
    Low-level: context-based parameter sharing PPO policy
    """

    def __init__(self, env_config):
        BaseFootballHierarchicalEnv.__init__(self, env_config)

        # high-level obs and act spaces
        self.high_level_observation_space = gym.spaces.Tuple(
            [gym.spaces.Box(low=-np.inf, high=np.inf, shape=(115,)) for _ in range(self.num_agents)])
        if self.context_type == "continuous":
            self.high_level_action_space = gym.spaces.Tuple(
                [gym.spaces.Box(low=-1.0, high=1.0, shape=(self.context_size,)) for _ in range(self.num_agents)])
        elif self.context_type == "discrete":
            self.high_level_action_space = gym.spaces.Tuple(
                [gym.spaces.Discrete(self.context_size) for _ in range(self.num_agents)])
        else:
            raise NotImplementedError("Unsupported high-level action space.")

    @property
    def high_level_obs(self):
        return {"high_level_policy": tuple(self.env_obs)}

    def high_level_actions(self, action_dict):
        return list(action_dict["high_level_policy"])

    @property
    def high_level_rewards(self):
        return {"high_level_policy": sum(self.low_level_accumulated_rew)}

    @property
    def high_level_infos(self):
        rewards = {i: rew for i, rew in enumerate(self.low_level_accumulated_rew)}
        self.env_info["rewards"] = rewards
        return {"high_level_policy": self.env_info}
