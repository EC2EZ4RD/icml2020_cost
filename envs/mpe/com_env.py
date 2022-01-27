import gym
import numpy as np

from .base_env import BaseMPE


class ComMPE(BaseMPE):
    """Wraps OpenAI Multi-Agent Particle env to be compatible with RLlib multi-agent communication."""

    def __init__(self, env_config: dict):
        BaseMPE.__init__(self, env_config)
        self.observation_space = gym.spaces.Tuple(
            [self._env.observation_space[0] for _ in range(self.num_agents)])
        self.action_space = gym.spaces.Tuple(
            [self._env.action_space[0] for _ in range(self.num_agents)])

    def reset(self, config=None):
        obs_list = BaseMPE.reset(self, config=self.env_config if config is None else config)
        return self.convert_obs(obs_list)

    def step(self, actions):
        obs_list, rew_list, done, info_list = BaseMPE.step(self, list(actions))
        rewards = {i: rew for i, rew in enumerate(rew_list)}
        info = {"rewards": rewards}
        if done:
            info["cover_rate"] = info_list[0]["cover_rate"]
        all_rewards = sum(rewards.values())

        return self.convert_obs(obs_list), all_rewards, done, info

    def convert_obs(self, obs_list):
        return tuple([np.array(obs, dtype=np.float32) for obs in obs_list])
