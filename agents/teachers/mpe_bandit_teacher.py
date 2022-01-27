import ray
import numpy as np

from .bandit_teacher import BanditTeacher, EXP3


class ExpertEXP3(EXP3):
    def __init__(self, gamma, num_task, update_interval):
        EXP3.__init__(self, gamma, num_task, update_interval)
        self._log_weights = np.array([0.5] + [0] * (num_task - 1))

    def update_train_reward(self, reward):
        """Get the episodic reward of a task."""
        reward = min(30, max(reward + 30, 0)) / 30
        self.train_reward_history.append(reward)

    def update_eval_reward(self, reward):
        self.eval_reward = min(30, max(reward/100 + 30, 0)) / 30
        task_i = self.sample
        tmp_reward = 0.6 * self.eval_reward + 0.4 * sum(self.train_reward_history) / len(self.train_reward_history)
        reward_corrected = tmp_reward / self.task_probabilities[task_i]
        self._log_weights[task_i] += self._gamma * reward_corrected / self._num_tasks


@ray.remote
class MPEContextualBanditTeacher(BanditTeacher):
    def __init__(self, config: dict, seed):
        assert config["env"] == "mpe"
        self.config = config
        self._seed = seed
        np.random.seed(self._seed)
        assert config["scenario"] == "curriculum_simple_spread_po" or config["scenario"] == "curriculum_push_ball_po"
        self._tasks = config["num_agent_candidates"]
        self._num_tasks = len(self._tasks)
        self._num_contexts = config["num_contexts"]
        self.update_interval = config["update_interval"]
        self.algo = [ExpertEXP3(config["gamma"], self._num_tasks, self.update_interval) for _ in range(self._num_contexts)]
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

    def update_eval_reward(self, reward):
        self.eval_reward = min(30, max(reward + 20, 0)) / 30
        task_i = self._tasks.index(self.sample)
        tmp_reward = 0.8 * self.eval_reward + 0.2 * sum(self.train_reward_history) / len(self.train_reward_history)
        # tmp_reward = sum(self.train_reward_history) / len(self.train_reward_history)
        reward_corrected = tmp_reward / self.task_probabilities[task_i]
        self._log_weights[task_i] += self._gamma * reward_corrected / self._n_tasks

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
