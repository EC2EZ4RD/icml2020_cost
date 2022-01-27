import ray
import numpy as np
from collections import deque

TASK_11_vs_11_easy_stochastic = {
    "name": "11_vs_11_easy_stochastic",
    "game_duration": 3000,
    "left_team_difficulty": 0.05,
    "right_team_difficulty": 0.05,
    "deterministic": False,
    "end_episode_on_score": False,
    "end_episode_on_out_of_play": False,
    "end_episode_on_possession_change": False,
    "ball_pos": [0.02, 0.0],
    "num_teammates": 10,
    "teammate_pos": [[0.000000, 0.020000],[0.000000, -0.020000],[-0.422000, -0.19576],[-0.500000, -0.06356],[-0.500000, 0.063559],[-0.422000, 0.195760],[-0.184212, -0.10568],[-0.267574, 0.000000],[-0.184212, 0.105680],[-0.010000, -0.21610]],
    "num_opponents": 10,
    "opponent_pos": [[-0.050000, 0.000000],[-0.010000, 0.216102],[-0.422000, -0.19576],[-0.500000, -0.06356],[-0.500000, 0.063559],[-0.422000, 0.195760],[-0.184212, -0.10568],[-0.267574, 0.000000],[-0.184212, 0.105680],[-0.010000, -0.21610]],
}

TASK_8_vs_8 = {
    "name": "8_vs_8",
    "game_duration": 3000,
    "left_team_difficulty": 0.05,
    "right_team_difficulty": 0.05,
    "deterministic": False,
    "end_episode_on_score": False,
    "end_episode_on_out_of_play": False,
    "end_episode_on_possession_change": False,
    "ball_pos": [0.02, 0.0],
    "num_teammates": 7,
    "teammate_pos": [[0.0, 0.02], [0.0, -0.02], [-0.1, -0.1], [-0.1, 0.1],[-0.422000, -0.19576],[-0.500000, -0.06356],[-0.500000, 0.063559]],
    "num_opponents": 7,
    "opponent_pos": [[-0.04, 0.04], [-0.04, -0.04], [-0.1, -0.1], [-0.1, 0.1],[-0.422000, -0.19576],[-0.500000, -0.06356],[-0.500000, 0.063559]],
}

TASK_5_vs_5 = {
    "name": "5_vs_5",
    "game_duration": 3000,
    "left_team_difficulty": 0.05,
    "right_team_difficulty": 0.05,
    "deterministic": False,
    "end_episode_on_score": False,
    "end_episode_on_out_of_play": False,
    "end_episode_on_possession_change": False,
    "ball_pos": [0.02, 0.0],
    "num_teammates": 4,
    "teammate_pos": [[0.0, 0.02], [0.0, -0.02], [-0.1, -0.1], [-0.1, 0.1]],
    "num_opponents": 4,
    "opponent_pos": [[-0.04, 0.04], [-0.04, -0.04], [-0.1, -0.1], [-0.1, 0.1]],
}
TASK_4_vs_4 = {
    "name": "4_vs_4",
    "game_duration": 3000,
    "left_team_difficulty": 0.05,
    "right_team_difficulty": 0.05,
    "deterministic": False,
    "end_episode_on_score": False,
    "end_episode_on_out_of_play": False,
    "end_episode_on_possession_change": False,
    "ball_pos": [0.02, 0.0],
    "num_teammates": 3,
    "teammate_pos": [[0.0, 0.02], [0.0, -0.02], [-0.1, -0.1]],
    "num_opponents": 3,
    "opponent_pos": [[-0.04, 0.04], [-0.04, -0.04], [-0.1, -0.1]],
}
TASK_3_vs_3 = {
    "name": "3_vs_3",
    "game_duration": 3000,
    "left_team_difficulty": 0.05,
    "right_team_difficulty": 0.05,
    "deterministic": False,
    "end_episode_on_score": False,
    "end_episode_on_out_of_play": False,
    "end_episode_on_possession_change": False,
    "ball_pos": [0.02, 0.0],
    "num_teammates": 2,
    "teammate_pos": [[0.0, 0.02], [0.0, -0.02]],
    "num_opponents": 2,
    "opponent_pos": [[-0.04, 0.04], [-0.04, -0.04]],
}
TASK_academy_pass_and_shoot_with_keeper = {
    # academy_pass_and_shoot_with_keeper scenario with offsides
    "name": "academy_pass_and_shoot_with_keeper",
    "game_duration": 400,
    "left_team_difficulty": 0.05,
    "right_team_difficulty": 0.05,
    "deterministic": False,
    "end_episode_on_score": True,
    "end_episode_on_out_of_play": True,
    "end_episode_on_possession_change": True,
    "ball_pos": [0.7, -0.28],
    "num_teammates": 2,
    "teammate_pos": [[0.7, 0.0], [0.7, -0.3]],
    "num_opponents": 1,
    "opponent_pos": [[-0.75, 0.3]],
    "shaping_rewards": "passing"
}

TASK_academy_empty_goal_close = {  # academy_empty_goal_close scenario with offsides
    "name": "academy_empty_goal_close",
    "game_duration": 400,
    "left_team_difficulty": 0.05,
    "right_team_difficulty": 0.05,
    "deterministic": False,
    "end_episode_on_score": True,
    "end_episode_on_out_of_play": True,
    "end_episode_on_possession_change": True,
    "ball_pos": [0.77, 0.0],
    "num_teammates": 1,
    "teammate_pos": [[0.75, 0.0]],
    "num_opponents": 0,
    "opponent_pos": [],
}
TASK_academy_empty_goal = {  # academy_empty_goal scenario with offsides
    "name": "academy_empty_goal",
    "game_duration": 400,
    "left_team_difficulty": 0.05,
    "right_team_difficulty": 0.05,
    "deterministic": False,
    "end_episode_on_score": True,
    "end_episode_on_out_of_play": True,
    "end_episode_on_possession_change": True,
    "ball_pos": [0.02, 0.0],
    "num_teammates": 1,
    "teammate_pos": [[0.0, 0.0]],
    "num_opponents": 0,
    "opponent_pos": [],
}
TASK_academy_3_vs_1 = {  # academy_3_vs_1 scenario with offsides
    "name": "academy_3_vs_1",
    "game_duration": 400,
    "left_team_difficulty": 0.05,
    "right_team_difficulty": 0.05,
    "deterministic": False,
    "end_episode_on_score": True,
    "end_episode_on_out_of_play": True,
    "end_episode_on_possession_change": True,
    "ball_pos": [0.62, 0.0],
    "num_teammates": 3,
    "teammate_pos": [[0.6, 0.0], [0.7, 0.2], [0.7, -0.2]],
    "num_opponents": 1,
    "opponent_pos": [[-0.75, 0.0]],
    "shaping_rewards": "passing"
}
TASK_defense_scenario_1 = {  # defense scenario 1
    "name": "defense_scenario_1",
    "game_duration": 400,
    "left_team_difficulty": 0.05,
    "right_team_difficulty": 0.05,
    "deterministic": False,
    "end_episode_on_score": True,
    "end_episode_on_out_of_play": True,
    "end_episode_on_possession_change": True,
    "ball_pos": [-0.62, 0.0],
    "num_teammates": 2,
    "teammate_pos": [[-0.75, -0.2], [-0.75, 0.2]],
    "num_opponents": 3,
    "opponent_pos": [[0.6, 0.0], [0.7, 0.2], [0.7, -0.2]],
}
TASK_defense_scenario_2 = {  # defense scenario 2
    "name": "defense_scenario_2",
    "game_duration": 400,
    "left_team_difficulty": 0.05,
    "right_team_difficulty": 0.05,
    "deterministic": False,
    "end_episode_on_score": True,
    "end_episode_on_out_of_play": True,
    "end_episode_on_possession_change": True,
    "ball_pos": [-0.62, 0.0],
    "num_teammates": 3,
    "teammate_pos": [[-0.75, 0.0], [-0.75, -0.2], [-0.75, 0.2]],
    "num_opponents": 3,
    "opponent_pos": [[0.6, 0.0], [0.7, 0.2], [0.7, -0.2]],
}

@ray.remote
class TimeTeacher(object):
    """A parent bandit teacher class"""
    def __init__(self, config: dict, seed):
        self._tasks = [TASK_academy_empty_goal_close, TASK_academy_3_vs_1, TASK_academy_pass_and_shoot_with_keeper,
                       TASK_3_vs_3, TASK_5_vs_5, TASK_8_vs_8, TASK_11_vs_11_easy_stochastic]
        self._n_tasks = len(self._tasks)
        self.config = config
        self.timesteps = 0
        self.stop_timesteps = self.config.get("stop_timesteps", 100000000)
        self._seed = seed
        np.random.seed(self._seed)
        self.sample_flag = False

        self.global_env_steps = 0
        self.global_high_level_steps = 0

        self.info = {}

    def update_timesteps(self, timesteps):
        self.timesteps = timesteps

    def set_sample_flag(self, flag):
        """a flag to enable env to reset"""
        self.sample_flag = flag

    def get_sample_flag(self):
        return self.sample_flag

    def sample_task(self):
        return self._tasks[self.timesteps // (self.stop_timesteps // self._n_tasks)]

    def get_task(self):
        return self._tasks[self.timesteps // (self.stop_timesteps // self._n_tasks)]

    def update_info(self, new_info: dict):
        self.info.update(new_info)

    def get_info(self):
        return self.info

    def inc_env_steps(self):
        self.global_env_steps += 1

    def get_env_steps(self):
        return self.global_env_steps

    def inc_high_level_steps(self):
        self.global_high_level_steps += 1

    def get_high_level_steps(self):
        return self.global_high_level_steps