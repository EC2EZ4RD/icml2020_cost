import ray

from .shared_policy_env import MultiAgentMPE


class MultiAgentCurriculumMPE(MultiAgentMPE):
    def __init__(self, env_config):
        MultiAgentMPE.__init__(self, env_config)

    def reset(self):
        if not self.eval:
            teacher = ray.get_actor("teacher")
            sample_flag = ray.get(teacher.get_sample_flag.remote())
            if sample_flag:
                teacher.sample_task.remote()
                self.num_agents = ray.get(teacher.get_task.remote())
                self.env_config["num_agents"] = self.num_agents
                print("Sample task flag and num_agents:", sample_flag, self.num_agents)
                teacher.set_sample_flag.remote(False)
        obs = MultiAgentMPE.reset(self, config=self.env_config)
        if not self.eval:
            self.observation_space = self._env.observation_space[0]
            self.action_space = self._env.action_space[0]
        return obs
