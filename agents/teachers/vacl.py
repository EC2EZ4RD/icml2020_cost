"""
Variational Automatic Curriculum Learning for Sparse-Reward Cooperative Multi-Agent Problems (VACL)

Paper: https://arxiv.org/abs/2111.04613
Code: This implementation borrows code from https://github.com/jiayu-ch15/Variational-Automatic-Curriculum-Learning

The issues of VACL algorithm:
    1) There are two fixed thresholds to split task space, which could directly influence the gradients.
        For example, there will be no gradients to update if the upper threshold is too high.
    2) Entity Progression assumes solved tasks will not degrade to unsolvable tasks,
        which could suffer from catastrophic forgetting
    3) The VACL teacher in fact does not take into account the feedbacks from solved tasks.
        Also, the proposed Entity Progression for discrete parameters is merely incremental.
    4) No explicit reward signals from the target task.
    5) Not combining shaping rewards to deal with the sparse reward problem.
"""

import ray
import numpy as np
import math
import random
from scipy.spatial.distance import cdist
import copy


def sort_by_novelty(list1, list2, topk: int):
    """Compute distance between each pair of the two lists using Euclidean distance (2-norm).
    The novelty is measured by the sum of the top-K distance.

    Returns:
        The given list sorted by novelty.
    """
    dist = cdist(np.array(list1).reshape(len(list1), -1), np.array(list2).reshape(len(list2), -1), metric='euclidean')
    if len(list2) < topk + 1:
        dist_k = dist
        novelty = np.sum(dist_k, axis=1) / len(list2)
    else:
        dist_k = np.partition(dist, kth=topk + 1, axis=1)[:, 0:topk + 1]
        novelty = np.sum(dist_k, axis=1) / topk

    zipped = zip(list1, novelty)
    sorted_zipped = sorted(zipped, key=lambda x: (x[1], np.mean(x[0])))
    result = zip(*sorted_zipped)
    return [list(x) for x in result][0]


def gradient_of_state(state, buffer, h, use_rbf=True):
    """Compute the gradient of given state w.r.t. the buffer."""
    gradient = np.zeros(state.shape)
    for buffer_state in buffer:
        if use_rbf:
            dist0 = state - np.array(buffer_state).reshape(-1)
            gradient += 2 * dist0 * np.exp(-dist0 ** 2 / h) / h
        else:
            gradient += 2 * (state - np.array(buffer_state).reshape(-1))
    norm = np.linalg.norm(gradient, ord=2)
    if norm > 0.0:
        gradient = gradient / np.linalg.norm(gradient, ord=2)
        gradient_zero = False
    else:
        gradient_zero = True
    return gradient, gradient_zero


@ray.remote
class VACLTeacher:
    """Create a VACL teacher instance with its node buffer w/o Entity Progression.

    Attributes:
        buffer_length: max length of the buffer.
        reproduction_num: # of newly added tasks within each episode
        Rmin: highest value threshold for active task space.
        Rmax: lowest value threshold for active task space.
        topk: used in get_novelty().
        legal_region:
    """

    def __init__(self, config, seed):
        random.seed(seed)
        np.random.seed(seed)
        self.config = copy.deepcopy(config)
        self.num_workers = self.config["num_workers"]
        self.num_envs_per_worker = self.config["num_envs_per_worker"]
        self.num_envs = self.num_envs_per_worker * self.num_workers
        self.solved_prop = self.config["solved_prop"]

        self.env_name = self.config["env"]
        self.scenario_name = self.config["scenario"]
        self.num_agents_candidates = self.config["num_agents_candidates"]
        self.buffer_length = self.config["buffer_length"]

        # gradient step hyperparameters
        self.reproduction_num = self.config["reproduction_num"]
        self.epsilon = self.config["epsilon"]
        self.delta = self.config["delta"]
        self.h = self.config["h"]

        # VACL hyperparameters
        self.Rmin = self.config["Rmin"]
        self.Rmax = self.config["Rmax"]
        self.del_switch = self.config["del_switch"]
        self.topk = self.config["topk"]
        self.num_initial_tasks = self.config["num_initial_tasks"]

        self.active_task_buffer = [[1] + [0] * (len(self.num_agents_candidates) - 1)] * self.num_initial_tasks
        self.active_task_buffer = sort_by_novelty(self.active_task_buffer, self.active_task_buffer, self.topk)
        self.all_solved_task_buffer = []
        self.newly_solved_tasks = []
        self.solved_tasks_indices = []
        self.active_tasks_indices = []
        self.current_samples = None
        self.num_sampled_active_tasks = 0
        self.stats = {}

        self.sample_tasks()
        self.env_count = 0
        self.training_scores = []  # scores of tasks with next num_agents

    def compute_gradient(self, use_gradient_noise=True):
        """Compute gradients and add new tasks to buffer.

        Args:
            use_gradient_noise: for exploration purpose.
        """
        if self.newly_solved_tasks:  # not all sampled tasks unsolved in the last episode
            for _ in range(self.reproduction_num):
                for task in self.newly_solved_tasks:
                    gradient, gradient_zero = gradient_of_state(np.array(task).reshape(-1), self.all_solved_task_buffer,
                                                                self.h)
                    assert len(task) == len(self.num_agents_candidates)
                    probs = copy.deepcopy(task)
                    for i in range(len(task)):
                        # execute gradient step
                        if not gradient_zero:
                            probs[i] += gradient[i] * self.epsilon
                        else:
                            probs[i] += -2 * self.epsilon * random.random() + self.epsilon
                        # rejection sampling
                        if use_gradient_noise:
                            probs[i] += -2 * self.delta * random.random() + self.delta
                        # softmax
                    probs = list(np.exp(probs) / np.sum(np.exp(probs)))
                    self.active_task_buffer.append(probs)

    def sample_tasks(self):
        """Sample tasks from node buffer after sampling gradients."""
        num_solved_samples = math.ceil(self.num_envs * self.solved_prop)  # solved tasks added to avoid forgetting
        num_active_samples = self.num_envs - num_solved_samples  # num of active tasks

        solved_tasks_indices = random.sample(range(len(self.all_solved_task_buffer)),
                                             k=min(len(self.all_solved_task_buffer), num_solved_samples))
        active_tasks_indices = random.sample(range(len(self.active_task_buffer)),
                                             k=min(len(self.active_task_buffer),
                                                   num_active_samples + num_solved_samples - len(solved_tasks_indices)))
        if len(active_tasks_indices) < num_active_samples:
            solved_tasks_indices = random.sample(range(len(self.all_solved_task_buffer)),
                                                 k=min(len(self.all_solved_task_buffer),
                                                       num_active_samples + num_solved_samples - len(
                                                           active_tasks_indices)))

        self.active_tasks_indices = np.sort(active_tasks_indices)
        self.solved_tasks_indices = np.sort(solved_tasks_indices)

        self.current_samples = [self.active_task_buffer[i] for i in active_tasks_indices] + \
                               [self.all_solved_task_buffer[i] for i in solved_tasks_indices]
        self.num_sampled_active_tasks = len(active_tasks_indices)

    def get_task(self):
        sampled_probs = self.current_samples[self.env_count]
        num_agents = np.random.choice(self.num_agents_candidates, p=sampled_probs)
        # print("VACL:", num_agents, sampled_probs)
        is_solved = self.env_count >= self.num_sampled_active_tasks
        self.env_count = (self.env_count + 1) % self.num_envs
        return num_agents, is_solved

    def send_score(self, episode_end_info):
        if not episode_end_info["solved"]:
            self.training_scores.append(episode_end_info["cover_rate"])

    def update_buffer(self):
        """Update the node buffer. VACL teacher only collects scores of active tasks.
            1) Task Expansion:
                a. add solved tasks to all_solved_task_buffer, and delete them from active_task_buffer.
                b. delete non-active tasks if active_task_buffer is full.
            2) Maintain buffer size with given criteria.

        Returns:
            statistics.
        """

        total_del_num = 0
        del_easy_num = 0
        del_hard_num = 0

        # task expansion
        self.newly_solved_tasks = []
        # print("VACL:", self.num_sampled_active_tasks, len(self.training_scores))
        for i, score in enumerate(self.training_scores):
            if score > self.Rmax:  # solved
                self.newly_solved_tasks.append(copy.deepcopy(self.active_task_buffer[
                                                            self.active_tasks_indices[i] - total_del_num]))
                del self.active_task_buffer[self.active_tasks_indices[i] - total_del_num]
                total_del_num += 1
                del_easy_num += 1
            elif score < self.Rmin:  # unsolved and buffer is full
                if len(self.active_task_buffer) >= self.buffer_length:
                    del self.active_task_buffer[self.active_tasks_indices[i] - total_del_num]
                    total_del_num += 1
                    del_hard_num += 1

        # maintain buffer size
        self.all_solved_task_buffer += self.newly_solved_tasks
        if len(self.active_task_buffer) > self.buffer_length:
            if self.del_switch == 'novelty':  # novelty deletion (for diversity)
                self.active_task_buffer = sort_by_novelty(
                    self.active_task_buffer, self.active_task_buffer, self.topk)[
                                          len(self.active_task_buffer) - self.buffer_length:]
            elif self.del_switch == 'random':  # random deletion
                del_num = len(self.active_task_buffer) - self.buffer_length
                del_index = random.sample(range(len(self.active_task_buffer)), del_num)
                del_index = np.sort(del_index)
                total_del_num = 0
                for i in range(del_num):
                    del self.active_task_buffer[del_index[i] - total_del_num]
                    total_del_num += 1
            else:  # FIFO queue deletion
                self.active_task_buffer = self.active_task_buffer[len(self.active_task_buffer) - self.buffer_length:]
        if len(self.all_solved_task_buffer) > self.buffer_length:
            self.all_solved_task_buffer = self.all_solved_task_buffer[
                                     len(self.all_solved_task_buffer) - self.buffer_length:]

        self.stats = {
            "len_active_task_buffer": len(self.active_task_buffer),
            "num_newly_solved_tasks": len(self.newly_solved_tasks),
            "del_easy_num": del_easy_num,
            "del_hard_num": del_hard_num,
            "num_sampled_active_tasks": self.num_sampled_active_tasks
        }
        self.training_scores = []

    def get_stats(self):
        return self.stats


@ray.remote
class VACLEPTeacher:
    """Create a VACL teacher w/ Entity Progression."""

    def __init__(self, config):
        self.num_envs_per_worker = config["num_envs_per_worker"]
        self.num_workers = config["num_workers"]
        self.num_envs = self.num_workers * self.num_envs_per_worker

        self.cur_num_agents = config["num_agent_candidates"][0]
        self.next_num_agents = None
        self.target_num_agents = config["num_agent_candidates"][-1]

        self.threshold_next = config["threshold_next"]
        self.mix_training = False  # set to True when EP starts

        self.decay_episodes = 0
        self.decay_interval = config["decay_interval"]

        self.init_ratio_next = config["ratio_next"]
        self.ratio_next = self.init_ratio_next

        self.env_count = 0
        self.training_scores = []  # scores of tasks with next num_agents

    def get_task(self):
        if self.mix_training:
            next_num_envs = math.ceil(self.num_envs * self.ratio_next)  # task prop of new num_agents
            cur_num_envs = self.num_envs - next_num_envs  # task prop of old num_agents
        else:
            cur_num_envs = self.num_envs
        if self.env_count < cur_num_envs:
            num_agents = self.cur_num_agents
        else:
            num_agents = self.next_num_agents
        self.env_count = (self.env_count + 1) % self.num_envs
        return num_agents

    def send_score(self, episode_end_info):
        if self.mix_training:
            if episode_end_info["num_agents"] == self.next_num_agents:
                self.training_scores.append(episode_end_info["cover_rate"])
        else:
            self.training_scores.append(episode_end_info["cover_rate"])

    def update(self):
        print("VACL-EP:", self.cur_num_agents, self.next_num_agents, np.mean(self.training_scores),
              len(self.training_scores), self.mix_training, self.decay_episodes, self.ratio_next)
        if self.mix_training:
            self.decay_episodes += 1
            # smoothly increase num_agents for EP
            if self.decay_episodes >= self.decay_interval:
                self.decay_episodes = 0
                self.ratio_next = min(self.ratio_next + 0.1, 1.0)
                if np.abs(self.ratio_next - 1.0) <= 1e-5:
                    self.mix_training = False

        if np.mean(self.training_scores) >= self.threshold_next:  # threshold for EP
            if self.cur_num_agents < self.target_num_agents:
                self.mix_training = True
                self.ratio_next = self.init_ratio_next
                if self.next_num_agents:  # not first EP
                    self.cur_num_agents = self.next_num_agents
                self.next_num_agents = min(self.cur_num_agents * 2,
                                           self.target_num_agents)  # increase num_agents exponentially
            if self.mix_training and self.cur_num_agents >= self.target_num_agents:  # EP completes
                self.mix_training = False

        self.training_scores = []
