import ray
import gym
from ray.rllib.utils.spaces.repeated import Repeated

from .shared_policy_env import MultiAgentMPE
from .com_env import ComMPE


class VACLMultiAgentCurriculumMPE(MultiAgentMPE):
    def __init__(self, env_config):
        MultiAgentMPE.__init__(self, env_config)
        self.solved = None

    def reset(self):
        # query new num_agents
        if not self.eval:
            teacher = ray.get_actor("teacher")
            self.num_agents, self.solved = ray.get(teacher.get_task.remote())
            self.env_config["num_agents"] = self.num_agents
        obs = MultiAgentMPE.reset(self, config=self.env_config)
        if not self.eval:
            self.observation_space = self._env.observation_space[0]
            self.action_space = self._env.action_space[0]
        return obs

    def step(self, action_dict):
        obs_dict, rew_dict, done_dict, info_dict = MultiAgentMPE.step(self, action_dict)
        if not self.eval and done_dict["__all__"]:
            teacher = ray.get_actor("teacher")
            teacher.send_score.remote({"solved": self.solved, "cover_rate": info_dict["agent_0"]["cover_rate"]})

        return obs_dict, rew_dict, done_dict, info_dict


class VACLEPMultiAgentCurriculumMPE(MultiAgentMPE):
    def __init__(self, env_config):
        MultiAgentMPE.__init__(self, env_config)

    def reset(self):
        # query new num_agents
        if not self.eval:
            teacher = ray.get_actor("teacher")
            self.num_agents = ray.get(teacher.get_task.remote())
            self.env_config["num_agents"] = self.num_agents
        obs = MultiAgentMPE.reset(self, config=self.env_config)
        if not self.eval:
            self.observation_space = self._env.observation_space[0]
            self.action_space = self._env.action_space[0]
        return obs

    def step(self, action_dict):
        obs_dict, rew_dict, done_dict, info_dict = MultiAgentMPE.step(self, action_dict)
        if not self.eval and done_dict["__all__"]:
            teacher = ray.get_actor("teacher")
            teacher.send_score.remote(info_dict["agent_0"])

        return obs_dict, rew_dict, done_dict, info_dict


class VACLComCurriculumMPE(ComMPE):
    def __init__(self, env_config):
        ComMPE.__init__(self, env_config)
        self.max_num_agents = self.env_config["target_num_agents"]
        self.observation_space = Repeated(
            self._env.observation_space[0], max_len=self.env_config["target_num_agents"])
        self.action_space = gym.spaces.Tuple(
            [self._env.action_space[0] for _ in range(self.env_config["target_num_agents"])])
        self.solved = None

    def reset(self):
        # query new num_agents
        if not self.eval:
            teacher = ray.get_actor("teacher")
            self.num_agents, self.solved = ray.get(teacher.get_task.remote())
            self.env_config["num_agents"] = self.num_agents
        obs_list = ComMPE.reset(self, config=self.env_config)
        return list(obs_list)[:self.num_agents]

    def step(self, actions):
        act = list(actions)[:self.num_agents]
        obs_list, all_rewards, done, info = ComMPE.step(self, list(act))
        info["num_agents"] = self.num_agents
        info["rewards"].update({i: 0.0 for i in range(self.num_agents, self.max_num_agents)})
        if not self.eval and done:
            teacher = ray.get_actor("teacher")
            teacher.send_score.remote({"solved": self.solved, "cover_rate": info["cover_rate"]})
        return list(obs_list), all_rewards, done, info
