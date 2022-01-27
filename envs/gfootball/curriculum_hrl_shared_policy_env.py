from .hrl_shared_policy_env import FootballHierarchicalMultiAgentEnv
from .curriculum_base_env import BaseFootballCurriculumEnv


class FootballCurriculumHierarchicalMultiAgentEnv(FootballHierarchicalMultiAgentEnv, BaseFootballCurriculumEnv):

    def __init__(self, env_config):
        self.task = None
        FootballHierarchicalMultiAgentEnv.__init__(self, env_config)

    def reset(self):
        self.env_obs = BaseFootballCurriculumEnv.reset(self)
        self.env_rewards = [0.0] * self.num_agents
        self.low_level_accumulated_rew = {i: 0.0 for i in range(self.num_agents)}
        return self.high_level_obs
