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
    "shaping_rewards": ["passing", "shooting"],
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
    "shaping_rewards": ["passing", "shooting"]
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
    "shaping_rewards": ["checkpoints", "shooting"]
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

class BanditTeacher(object):
    """A parent bandit teacher class"""

    def __init__(self, config: dict, seed):
        self._tasks = getattr(self, '_tasks',  [TASK_5_vs_5])
        self._n_tasks = len(self._tasks)
        self.config = config
        self._gamma = config["gamma"]
        self._seed = seed
        np.random.seed(self._seed)
        self.iter_count = 0
        self.update_interval = config["update_interval"]
        self._log_weights = np.zeros(self._n_tasks)
        self.sample = None
        self.train_reward_history = list()
        self.eval_reward_history = list()
        self.context_history = list()
        self.eval_reward = 0
        self.sample_task()
        self.sample_flag = False


        self.global_env_steps = 0
        self.global_high_level_steps = 0

        self.info = {}

    @property
    def task_probabilities(self):
        weights = np.exp(self._log_weights - np.sum(self._log_weights))
        probs = ((1 - self._gamma) * weights / np.sum(weights) + self._gamma / self._n_tasks)
        return probs

    def set_sample_flag(self, flag):
        """a flag to enable env to reset"""
        self.sample_flag = flag

    def get_sample_flag(self):
        return self.sample_flag

    def sample_task(self):
        """Samples a task, according to current Exp3 belief."""
        self.sample = self._tasks[np.random.choice(self._n_tasks, p=self.task_probabilities)]

    def update_train_reward(self, reward):
        """Get the episodic reward of a task."""
        reward = min(30, max(reward + 20, 0)) / 30
        self.train_reward_history.append(reward)

    def update_eval_reward(self, reward):
        self.eval_reward = min(30, max(reward + 20, 0)) / 30
        task_i = self._tasks.index(self.sample)
        tmp_reward = 0.8 * self.eval_reward + 0.2 * sum(self.train_reward_history) / len(self.train_reward_history)
        reward_corrected = tmp_reward / self.task_probabilities[task_i]
        self._log_weights[task_i] += self._gamma * reward_corrected / self._n_tasks

    def get_probs(self):
        return self.task_probabilities

    def get_task(self):
        return self.sample

    def update_context(self, context):
        self.context_history.append(context)

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


@ray.remote
class DiscreteTaskBanditTeacher(BanditTeacher):
    def __init__(self, config: dict, seed):
        self._tasks = [TASK_5_vs_5, TASK_4_vs_4, TASK_3_vs_3,
                       TASK_academy_pass_and_shoot_with_keeper, TASK_academy_empty_goal, TASK_academy_empty_goal_close,
                       TASK_academy_3_vs_1, TASK_defense_scenario_1, TASK_defense_scenario_2]
        super().__init__(config, seed)


@ray.remote
class SimpleTaskBanditTeacher(BanditTeacher):
    def __init__(self, config: dict, seed):
        self._tasks = [TASK_5_vs_5, TASK_3_vs_3,
                       TASK_academy_pass_and_shoot_with_keeper,
                       TASK_academy_3_vs_1]
        super().__init__(config, seed)


@ray.remote
class TaskwiseBanditTeacher(BanditTeacher):
    def __init__(self, config: dict, seed):
        self._tasks = [TASK_5_vs_5, TASK_3_vs_3,
                       TASK_academy_pass_and_shoot_with_keeper,
                       TASK_academy_3_vs_1]
        super().__init__(config, seed)
        self._n_tasks = len(self._tasks)

        self.current_task_index = None
        self.reward_history = [deque([0] * self.update_interval, maxlen=self.update_interval) for _ in
                               range(self._n_tasks)]
        self.eval_reward = -10
        self.sample_task()  # after this funciton, self.sample is current task

        self.global_env_steps = 0
        self.global_high_level_steps = 0

        self.info = {}

    def update_train_reward(self, reward):
        """Get the episodic reward of a task."""
        reward = min(20, max(reward + 20, 0)) / 20
        self.reward_history[self.current_task_index].appendleft(reward)

    def update_train_task_reward(self, reward):
        """Get the episodic reward of a task."""
        reward = min(20, max(reward + 20, 0)) / 20
        self.reward_history[self.current_task_index].appendleft(reward)

    def update_eval_reward(self, reward):
        self.eval_reward = min(20, max(reward + 20, 0)) / 20  # range [0,1]
        task_i = self._tasks.index(self.sample)
        assert task_i == self.current_task_index

        # mean_on_current_task = mean(self.reward_history[task_i])
        var_on_current_task = np.var(self.reward_history[task_i])  # range [0,0.25]
        tmp_reward = 0.4 * self.eval_reward + \
                     0.4 * sum(self.reward_history[task_i]) / len(self.reward_history[task_i]) + \
                     0.2 * var_on_current_task
        reward_corrected = tmp_reward / self.task_probabilities[task_i]
        self._log_weights[task_i] += self._gamma * reward_corrected / self._n_tasks

@ray.remote
class EvalOn5v5BanditTeacher(BanditTeacher):
    """A bandit teacher that controls the env tasks."""

    def __init__(self, config: dict, seed):
        self._tasks = [TASK_5_vs_5, TASK_3_vs_3,
                       TASK_academy_pass_and_shoot_with_keeper,
                       TASK_academy_3_vs_1]
        super().__init__(config, seed)

    def update_eval_reward(self, reward):
        self.eval_reward = min(30, max(reward + 20, 0)) / 30
        task_i = self._tasks.index(self.sample)
        tmp_reward = self.eval_reward
        reward_corrected = tmp_reward / self.task_probabilities[task_i]
        self._log_weights[task_i] += self._gamma * reward_corrected / self._n_tasks

class EXP3(object):
    def __init__(self, gamma, num_task, update_interval):
        self._num_tasks = num_task
        self._gamma = gamma
        self._log_weights = np.zeros(self._num_tasks)
        self.update_interval = 5*update_interval
        self.train_reward_history = deque([0] * self.update_interval, maxlen=self.update_interval)
        self.eval_reward_history = list()
        self.sample = 0

    @property
    def task_probabilities(self):
        weights = np.exp(self._log_weights - np.sum(self._log_weights))
        probs = ((1 - self._gamma) * weights / np.sum(weights) + self._gamma / self._num_tasks)
        return probs

    def sample_task(self):
        """Samples a task, according to current Exp3 belief."""
        self.sample = np.random.choice(self._num_tasks, p=self.task_probabilities)
        return self.sample

    def update_train_reward(self, reward):
        """Get the episodic reward of a task."""
        reward = min(30, max(reward + 20, 0)) / 30
        self.train_reward_history.append(reward)

    def update_eval_reward(self, reward):
        self.eval_reward = min(30, max(reward + 20, 0)) / 30
        task_i = self.sample
        tmp_reward = 0.6 * self.eval_reward + 0.4 * sum(self.train_reward_history) / len(self.train_reward_history)
        reward_corrected = tmp_reward / self.task_probabilities[task_i]
        self._log_weights[task_i] += self._gamma * reward_corrected / self._num_tasks



@ray.remote
class ContextualBanditTeacher(BanditTeacher):
    """A parent bandit teacher class"""

    def __init__(self, config: dict, seed):
        self.config = config
        self._seed = seed
        np.random.seed(self._seed)
        self._tasks = [TASK_11_vs_11_easy_stochastic, TASK_8_vs_8, TASK_5_vs_5, TASK_3_vs_3,
                       TASK_academy_pass_and_shoot_with_keeper,
                       TASK_academy_3_vs_1, TASK_academy_empty_goal_close]
        self._num_tasks = len(self._tasks)
        self._num_contexts = config["num_contexts"]
        self.update_interval = config["update_interval"]
        self.algo = [EXP3(config["gamma"], self._num_tasks, self.update_interval) for _ in range(self._num_contexts)]
        from sklearn.cluster import Birch
        self.context_classifier = Birch(n_clusters=self._num_contexts)
        self.context_class = 0
        self.sample = np.random.choice(self._tasks)
        self.context_history = list()
        self.sample_flag = False
        self.global_env_steps = 0
        self.global_high_level_steps = 0
        self.info = {}

    def update_context_reward(self, context, reward, train=True):
        self.context_history.append(context)
        if len(self.context_history) < 2 * self._num_contexts:
            self.sample = np.random.choice(self._tasks)
            self.context_class = len(self.context_history) % self._num_contexts
        elif len(self.context_history) == 2 * self._num_contexts:
            self.context_classifier.partial_fit(list(self.context_history))
            self.sample = np.random.choice(self._tasks)
            self.context_class = len(self.context_history) % self._num_contexts
        else:
            self.context_history.pop(0)
            self.context_classifier.partial_fit([context])
            context_class = self.context_classifier.predict([context])[0]
            if train:
                self.algo[context_class].update_train_reward(reward)
            else:
                self.algo[context_class].update_eval_reward(reward)
            self.sample = self._tasks[self.algo[context_class].sample_task()]
            self.context_class = context_class

    def sample_task(self):
        return self.sample

    def set_sample_flag(self, flag):
        """a flag to enable env to reset"""
        self.sample_flag = flag

    def get_sample_flag(self):
        return self.sample_flag

    def get_probs(self):
        return self.algo[self.context_class].task_probabilities

    def get_task(self):
        return self.sample

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