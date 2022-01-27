import gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from envs.mpe.base_env import BaseMPE


class MultiAgentMPE(BaseMPE, MultiAgentEnv):
    """Wraps OpenAI Multi-Agent Particle env to be compatible with RLlib multi-agent."""

    def __init__(self, env_config: dict):
        BaseMPE.__init__(self, env_config)

    def reset(self, config=None):
        return self._make_dict(BaseMPE.reset(self, config=self.env_config if config is None else config))

    def step(self, action_dict):
        actions = [action_dict[f"agent_{i}"] for i in range(self.num_agents)]
        obs_list, rew_list, done, info_list = BaseMPE.step(self, actions)
        obs_dict = self._make_dict(obs_list)
        rew_dict = self._make_dict(rew_list)
        info_dict = self._make_dict(info_list)
        done_dict = {"__all__": done}
        return obs_dict, rew_dict, done_dict, info_dict

    def _make_dict(self, values):
        return {f"agent_{i}": values[i] for i in range(self.num_agents)}


def test():
    for scenario_name in ["curriculum_simple_spread_po", "curriculum_push_ball_po"]:
        print("==========\nscenario_name: ", scenario_name)
        env_config = {"scenario_name": scenario_name, "num_agents": 3,
                      "num_landmarks": 3, "seed": 42,
                      "episode_length": 70, "num_observable_agents": 3,
                      "custom_configs": {"evaluation": False}}
        env = MultiAgentMPE(env_config)
        print("obs: ", env.reset(config=env_config))
        print(env.observation_space)
        print(env.action_space)

        for _ in range(100):
            ac_dict = {}
            for id in range(env.num_agents):
                sample = env.action_space.sample()
                if isinstance(env.action_space, gym.spaces.Discrete) or \
                        isinstance(env.action_space, gym.spaces.Box):
                    ac_dict[f"agent_{id}"] = sample
                elif isinstance(env.action_space, gym.spaces.MultiDiscrete):
                    # print("sample: ", sample)
                    # print("ac_space: ", env.action_space.nvec)
                    ac_dict[f"agent_{id}"] = np.zeros(sum(env.action_space.nvec))
                    start_ls = np.cumsum([0] + list(env.action_space.nvec))[:-1]
                    for l in list(start_ls + sample):
                        ac_dict[f"agent_{id}"][l] = 1.0
                else:
                    raise NotImplementedError
            env.render()
            print("action_dict: ", ac_dict)
            print(env.step(ac_dict))


if __name__ == "__main__":
    test()
