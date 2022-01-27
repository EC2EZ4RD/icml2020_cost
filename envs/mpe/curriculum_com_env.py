import ray
import gym
import numpy as np
from ray.rllib.utils.spaces.repeated import Repeated

from .com_env import ComMPE


class CurriculumComMPE(ComMPE):
    def __init__(self, env_config: dict):
        ComMPE.__init__(self, env_config)
        self.max_num_agents = self.env_config["target_num_agents"]
        self.observation_space = Repeated(
            self._env.observation_space[0], max_len=self.env_config["target_num_agents"])
        self.action_space = gym.spaces.Tuple(
            [self._env.action_space[0] for _ in range(self.env_config["target_num_agents"])])

    def reset(self, config=None):
        if not self.eval:
            teacher = ray.get_actor("teacher")
            sample_flag = ray.get(teacher.get_sample_flag.remote())
            if sample_flag:
                teacher.sample_task.remote()
                self.num_agents = ray.get(teacher.get_task.remote())
                self.env_config["num_agents"] = self.num_agents
                print("Sample task flag and num_agents:", sample_flag, self.num_agents)
                teacher.set_sample_flag.remote(False)
        obs_list = ComMPE.reset(self, config=self.env_config)
        return list(obs_list)[:self.num_agents]

    def step(self, actions):
        act = list(actions)[:self.num_agents]
        obs_list, all_rewards, done, info = ComMPE.step(self, list(act))
        info["num_agents"] = self.num_agents
        info["rewards"].update({i: 0.0 for i in range(self.num_agents, self.max_num_agents)})
        return list(obs_list), all_rewards, done, info
