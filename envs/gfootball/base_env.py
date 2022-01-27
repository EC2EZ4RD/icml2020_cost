import gym
import gfootball.env as football_env
import copy

from .wrappers import ObservationWrapper, CheckpointRewardWrapper, ShootRewardWrapper, PassRewardWrapper


class BaseFootballEnv(gym.Env):
    """Wraps Google Football env to be compatible RLlib."""

    def __init__(self, env_config):
        """Create a new multi-agent GFootball env compatible with RLlib.

        Args:
            env_config (dict): Arguments to pass to the underlying gfootball.env instance.
        """
        assert env_config["representation"] == "raw"
        self.env_config = copy.deepcopy(env_config)
        self._env = football_env.create_environment(
            **{k: v for k, v in self.env_config.items() if k not in ["custom_configs"]})
        self.eval = self.env_config["custom_configs"]["evaluation"]
        if self.eval:
            assert self.env_config["rewards"] == "scoring", self.env_config
        self.wrap_env()

        self.num_agents = self.env_config["number_of_left_players_agent_controls"]
        self.accumulated_score = 0

    def reset(self, **kwargs):
        """Reset the environment.

        Returns:
            obs_list (list): The initial observation.
        """
        (obs_list, raw_obs) = self._env.reset(**kwargs)
        assert self.num_agents == len(raw_obs)
        self.accumulated_score = 0

        return obs_list

    def step(self, actions):
        """Steps in the environment.

        Returns:
            obs_list (list): New observations for each ready agent.
            rew_list (list): Reward values for each ready agent.
            done (bool): Done values for each ready agent.
            info (dict): Optional info values for each agent.
        """
        (obs_list, raw_obs), rew_list, done, info = self._env.step(actions)
        self.accumulated_score += info["score_reward"]
        info = {"score": self.accumulated_score}
        return obs_list, rew_list, done, info

    def close(self):
        """Close the environment."""
        self._env.close()

    def render(self, mode="human"):
        self._env.render()

    def wrap_env(self):
        self._env = ObservationWrapper(self._env)
        shaping_rewards = self.env_config["custom_configs"].get("shaping_rewards")
        if not self.eval:
            if shaping_rewards is not None:
                if "shooting" in shaping_rewards:
                    self._env = ShootRewardWrapper(self._env)
                if "checkpoints" in shaping_rewards:
                    self._env = CheckpointRewardWrapper(self._env)
                if "passing" in shaping_rewards:
                    self._env = PassRewardWrapper(self._env)
        else:
            assert shaping_rewards is None, self.env_config
