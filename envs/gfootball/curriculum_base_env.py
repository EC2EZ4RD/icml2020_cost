import gfootball.env as football_env
import ray

from .wrappers import CheckpointRewardWrapper, ShootRewardWrapper, PassRewardWrapper
from .base_env import BaseFootballEnv
from agents.teachers.bandit_teacher import TASK_5_vs_5


class BaseFootballCurriculumEnv(BaseFootballEnv):
    """Wraps Google Football env to be compatible with curriculum learning."""

    def __init__(self, env_config):
        self.task = TASK_5_vs_5
        BaseFootballEnv.__init__(self, env_config)

    def reset(self):
        """Reset the environment and sample a new task from the teacher.

        Returns:
            obs_list (list): The initial observation.
        """
        if not self.eval:
            teacher = ray.get_actor("teacher")
            sample_flag = ray.get(teacher.get_sample_flag.remote())
            if sample_flag:
                teacher.sample_task.remote()
                self.task = ray.get(teacher.get_task.remote())
                print("Sample task:", sample_flag, self.task)
                teacher.set_sample_flag.remote(False)
                self.num_agents = self.task["num_teammates"]
                self.env_config["number_of_left_players_agent_controls"] = self.num_agents
                self.env_config["other_config_options"] = {"task": self.task}
                self.env_config["env_name"] = "curriculum"
                self._env.close()
                self._env = football_env.create_environment(**{
                    k: v for k, v in self.env_config.items() if k not in ["custom_configs"]})
                self.wrap_env()
        return BaseFootballEnv.reset(self, task=self.task)

    def wrap_env(self):
        BaseFootballEnv.wrap_env(self)
        if not self.eval:
            if self.task is not None and "shaping_rewards" in self.task:
                if "shooting" in self.task["shaping_rewards"]:
                    self._env = ShootRewardWrapper(self._env)
                if "checkpoints" in self.task["shaping_rewards"]:
                    self._env = CheckpointRewardWrapper(self._env)
                if "passing" in self.task["shaping_rewards"]:
                    self._env = PassRewardWrapper(self._env)
