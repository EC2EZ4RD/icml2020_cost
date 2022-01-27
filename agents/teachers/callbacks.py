from typing import Optional, Dict

import ray
from ray.rllib.env import BaseEnv
from ray.rllib.utils.typing import PolicyID
from ray.rllib.policy.policy import Policy
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode


class TeacherCallback(DefaultCallbacks):
    def on_train_result(self, *, trainer, result: dict, **kwargs) -> None:
        """Send train and eval reward to teacher at the end of Trainable.train().

        Args:
            trainer (Trainer): Current trainer instance.
            result (dict): Dict of results returned from trainer.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        # Collect episode mean reward to update the teacher.
        teacher = ray.get_actor("teacher")
        teacher.update_train_reward.remote(result["episode_reward_mean"])
        if "evaluation" in result:
            hist_stats = result["evaluation"]["hist_stats"]
            eval_episode_reward_mean = sum(hist_stats["episode_reward"]) / len(hist_stats["episode_reward"])
            teacher.update_eval_reward.remote(eval_episode_reward_mean)
            teacher.set_sample_flag.remote(True)

    def on_episode_end(self,
                       *,
                       worker: "RolloutWorker",
                       base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode,
                       env_index: Optional[int] = None,
                       **kwargs) -> None:
        """Collect task probabilities and score when an episode is done."""
        teacher = ray.get_actor("teacher")
        task_probs = ray.get(teacher.get_probs.remote())
        for i in range(len(task_probs)):
            # teacher.update_context.remote(policies["default_policy"])
            episode.custom_metrics[f"task_probs_{i + 1}_agents"] = task_probs[i]


class ContextualTeacherCallback(DefaultCallbacks):
    def on_train_result(self, *, trainer, result: dict, **kwargs) -> None:
        """Send train and eval reward to teacher at the end of Trainable.train().

        Args:
            trainer (Trainer): Current trainer instance.
            result (dict): Dict of results returned from trainer.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        # Collect episode mean reward to update the teacher.
        teacher = ray.get_actor("teacher")
        if trainer.get_policy("high_level_policy"):
            context = getattr(trainer.get_policy("high_level_policy").model, "last_hx", [0])
        elif trainer.get_policy():
            context = getattr(trainer.get_policy().model, "last_hx", [0])
        else:
            context = [0]
        teacher.update_context_reward.remote(context, result["episode_reward_mean"], True)
        if "evaluation" in result:
            hist_stats = result["evaluation"]["hist_stats"]
            eval_episode_reward_mean = sum(hist_stats["episode_reward"]) / len(hist_stats["episode_reward"])
            teacher.update_context_reward.remote(context, eval_episode_reward_mean, False)
            teacher.set_sample_flag.remote(True)

    def on_episode_end(self,
                       *,
                       worker: "RolloutWorker",
                       base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode,
                       env_index: Optional[int] = None,
                       **kwargs) -> None:
        """Collect task probabilities and score when an episode is done."""
        teacher = ray.get_actor("teacher")
        task_probs = ray.get(teacher.get_probs.remote())
        for i in range(len(task_probs)):
            episode.custom_metrics[f"task_probs_{i + 1}_agents"] = task_probs[i]


class TimeTeacherCallback(DefaultCallbacks):
    def on_train_result(self, *, trainer, result: dict, **kwargs) -> None:
        """Send train and eval reward to teacher at the end of Trainable.train().

        Args:
            trainer (Trainer): Current trainer instance.
            result (dict): Dict of results returned from trainer.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        # Collect episode mean reward to update the teacher.
        teacher = ray.get_actor("teacher")
        timesteps_total = result["timesteps_total"]
        teacher.update_timesteps.remote(timesteps_total)
        if "evaluation" in result:
            teacher.set_sample_flag.remote(True)


class TaskwiseTeacherCallback(DefaultCallbacks):
    def on_train_result(self, *, trainer, result: dict, **kwargs) -> None:
        """Send train and eval reward to teacher at the end of Trainable.train().

        Args:
            trainer (Trainer): Current trainer instance.
            result (dict): Dict of results returned from trainer.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        # Collect episode mean reward to update the teacher.
        teacher = ray.get_actor("teacher")
        teacher.update_train_task_reward.remote(result["episode_reward_mean"])
        if "evaluation" in result:
            hist_stats = result["evaluation"]["hist_stats"]
            eval_episode_reward_mean = sum(hist_stats["episode_reward"]) / len(hist_stats["episode_reward"])
            teacher.update_eval_reward.remote(eval_episode_reward_mean)
            teacher.set_sample_flag.remote(True)

    def on_episode_end(self,
                       *,
                       worker: "RolloutWorker",
                       base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode,
                       env_index: Optional[int] = None,
                       **kwargs) -> None:
        """Collect task probabilities and score when an episode is done."""
        teacher = ray.get_actor("teacher")
        task_probs = ray.get(teacher.get_probs.remote())
        for i in range(len(task_probs)):
            episode.custom_metrics[f"task_probs_{i + 1}_agents"] = task_probs[i]


class VACLMPETeacherCallback(DefaultCallbacks):
    """VACL w/o Entity Progression for MPE (teacher score: cover_rate)."""
    def on_episode_end(self,
                       *,
                       worker: "RolloutWorker",
                       base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode,
                       env_index: Optional[int] = None,
                       **kwargs) -> None:
        """Set task when an episode is done (before the next episode resets)."""
        if episode.last_info_for():
            cover_rate = episode.last_info_for()["cover_rate"]
            num_agents = episode.last_info_for()["num_agents"]
        else:
            cover_rate = episode.last_info_for("agent_0")["cover_rate"]
            num_agents = episode.last_info_for("agent_0")["num_agents"]
        episode.custom_metrics["cover_rate"] = cover_rate
        episode.custom_metrics["num_agents"] = num_agents

        # add VACL statistics
        teacher = ray.get_actor("teacher")
        stats = ray.get(teacher.get_stats.remote())
        for k, v in stats.items():
            episode.custom_metrics[k] = v

    def on_train_result(self, *, trainer, result: dict, **kwargs) -> None:
        """Send training scores to update VACL node buffer at the end of Trainable.train().

        Make sure rollout_fragment_length==episode_length and train_batch_size==num_envs*rollout_fragment_length,
        so that this update is conducted on episode end, i.e., the original implementation of VACL.
        """
        teacher = ray.get_actor("teacher")
        teacher.update_buffer.remote()
        teacher.compute_gradient.remote()
        teacher.sample_tasks.remote()


class VACLEPMPETeacherCallback(DefaultCallbacks):
    """VACL w/o Entity Progression for MPE (teacher score: cover_rate)."""
    def on_episode_end(self,
                       *,
                       worker: "RolloutWorker",
                       base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode,
                       env_index: Optional[int] = None,
                       **kwargs) -> None:
        """Collect custom metrics."""
        if episode.last_info_for():
            cover_rate = episode.last_info_for()["cover_rate"]
            num_agents = episode.last_info_for()["num_agents"]
        else:
            cover_rate = episode.last_info_for("agent_0")["cover_rate"]
            num_agents = episode.last_info_for("agent_0")["num_agents"]
        episode.custom_metrics["cover_rate"] = cover_rate
        episode.custom_metrics["num_agents"] = num_agents

    def on_train_result(self, *, trainer, result: dict, **kwargs) -> None:
        """Send training scores to update VACL node buffer at the end of Trainable.train().

        Make sure rollout_fragment_length==episode_length and train_batch_size==num_envs*rollout_fragment_length,
        so that this update is conducted on episode end, i.e., the original implementation of VACL.
        """
        teacher = ray.get_actor("teacher")
        teacher.update.remote()
