import gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from .base_env import BaseFootballEnv


class BaseFootballHierarchicalEnv(BaseFootballEnv, MultiAgentEnv):
    """Wraps Google Football env to be compatible with HRL."""

    def __init__(self, env_config):
        BaseFootballEnv.__init__(self, env_config)

        # HRL configs
        self.high_level_interval = self.env_config["custom_configs"]["high_level_interval"]
        self.context_type = self.env_config["custom_configs"]["context_type"]
        self.context_size = self.env_config["custom_configs"]["context_size"]

        # Book-keeping for high-low level transition
        self.context = None
        self.env_obs = None
        self.env_rewards = {i: 0.0 for i in range(self.num_agents)}
        self.env_info = {}
        
        # Book-keeping for high-level within a macro-step
        self.steps = 0
        self.low_level_accumulated_rew = {}

        # low-level obs and act spaces
        self.low_level_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(115 + self.context_size,))
        self.low_level_action_space = gym.spaces.Discrete(19)

    def reset(self):
        self.env_obs = BaseFootballEnv.reset(self)
        self.low_level_accumulated_rew = {i: 0.0 for i in range(self.num_agents)}
        return self.high_level_obs

    def step(self, action_dict):
        if list(action_dict.keys())[0].startswith("agent_"):
            return self._low_level_step(action_dict)
        else:
            return self._high_level_step(action_dict)

    def _high_level_step(self, action_dict):
        actions = self.high_level_actions(action_dict)
        if self.context_type == "discrete":
            # one-hot encoding
            one_hot_context = np.zeros((len(actions), self.context_size))
            for i, act in enumerate(actions):
                one_hot_context[i, act] = 1
            self.context = one_hot_context
        else:
            self.context = actions

        self.low_level_accumulated_rew = {i: 0.0 for i in range(self.num_agents)}
        obs = self.low_level_obs
        rew = self.low_level_rewards
        done = {"__all__": False}
        info = self.low_level_infos

        return obs, rew, done, info

    def _low_level_step(self, action_dict):
        self.env_obs, self.env_rewards, done, self.env_info = \
            BaseFootballEnv.step(self, self.low_level_actions(action_dict))
        for i in range(self.num_agents):
            self.low_level_accumulated_rew[i] += self.env_rewards[i]
        self.steps += 1
        self.env_info["num_agents"] = self.num_agents

        # Handle env termination & transitions back to higher level
        if done or self.steps == self.high_level_interval:
            self.steps = 0
            obs = self.high_level_obs
            reward = self.high_level_rewards
            infos = self.high_level_infos
        else:
            obs = self.low_level_obs
            reward = self.low_level_rewards
            infos = self.low_level_infos

        done = {"__all__": done}

        return obs, reward, done, infos
    
    @property
    def low_level_obs(self):
        return {f"agent_{i}": np.concatenate((self.env_obs[i], self.context[i])) for i in range(self.num_agents)}

    def low_level_actions(self, action_dict):
        return [action_dict[f"agent_{i}"] for i in range(self.num_agents)]

    @property
    def low_level_infos(self):
        return {f"agent_{i}": self.env_info for i in range(self.num_agents)}

    @property
    def low_level_rewards(self):
        return {f"agent_{i}": self.env_rewards[i] for i in range(self.num_agents)}

    @property
    def high_level_obs(self):
        raise NotImplementedError

    def high_level_actions(self, action_dict):
        raise NotImplementedError

    @property
    def high_level_rewards(self):
        raise NotImplementedError

    @property
    def high_level_infos(self):
        raise NotImplementedError
