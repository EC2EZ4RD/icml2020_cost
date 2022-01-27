import argparse
from typing import Optional, Dict
import yaml
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env import BaseEnv
from ray.rllib.utils.typing import PolicyID
from ray.rllib.policy.policy import Policy
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.agents.callbacks import MultiCallbacks, DefaultCallbacks

from envs.gfootball import FootballCurriculumMultiAgentEnv, eval_fn
from agents.teachers import TEACHER_LIST, CALLBACK_LIST

EXAMPLE_USAGE = """
python run_curriculum_shared_ppo_gfootball.py
"""


def create_args():
    parser = argparse.ArgumentParser(
        description="Train gfootball agents with parameter sharing PPO and curriculum learning.",
        epilog=EXAMPLE_USAGE)
    parser.add_argument(
        "-f",
        "--config-file",
        default="../configs/gfootball_shared_ppo_curriculum.yaml",
        type=str,
        help="Use config options from this yaml file.")
    parser.add_argument("--num-agents", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--env", type=str, choices=["gfootball"], default=None)
    parser.add_argument("--scenario", type=str, choices=["5_vs_5"], default=None)
    parser.add_argument(
        "--stop-timesteps",
        type=int,
        default=None,
        help="Number of timesteps to train.")
    parser.add_argument(
        "--from-checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint file for restoring a previously saved Trainer state.")
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="The experiment name")
    parser.add_argument("--num-eval-workers", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--teacher", type=str,
                        choices=["agent_num_bandit", "discrete_bandit", "simple_bandit", "task_wise_bandit",
                                 "eval_on_5v5_bandit"], default=None)

    args = parser.parse_args()
    with open(args.config_file) as f:
        configs = yaml.safe_load(f)

    # argparse has higher priority than yaml file
    for k, v in vars(args).items():
        if v is not None:
            configs[k] = v

    return argparse.Namespace(**configs)


class CustomMetricsCallback(DefaultCallbacks):
    def on_episode_end(self,
                       *,
                       worker: "RolloutWorker",
                       base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode,
                       env_index: Optional[int] = None,
                       **kwargs) -> None:
        """Runs when an episode is done."""
        episode.custom_metrics["score"] = episode.last_info_for("agent_0")["score"]


if __name__ == "__main__":
    args = create_args()
    ray.init()
    teacher = TEACHER_LIST[args.teacher].options(
        name="teacher", num_cpus=1, num_gpus=0).remote(config=args.teacher_config, seed=args.seed)

    assert args.env == "football"
    env_name = "gfootball_" + args.scenario
    register_env(env_name, lambda cfg: FootballCurriculumMultiAgentEnv(cfg))

    config = {
        "framework": "torch",
        "log_level": "INFO",
        "callbacks": MultiCallbacks([CALLBACK_LIST[args.teacher], CustomMetricsCallback]),
        "observation_filter": "NoFilter",
        "seed": args.seed,
        "batch_mode": "truncate_episodes",
        "explore": True,

        # === Resource Settings ===
        "num_workers": args.num_workers,
        "num_envs_per_worker": args.num_envs_per_worker,
        "num_cpus_per_worker": 1,
        "num_cpus_for_driver": 1,
        "num_gpus": args.num_gpus,
        "num_gpus_per_worker": 0,
        "env": env_name,
        "env_config": {
            "env_name": args.scenario,
            "stacked": False,
            "rewards": "scoring",
            "write_goal_dumps": False,
            "write_full_episode_dumps": True,
            "render": False,
            "write_video": False,
            "dump_frequency": 200,
            "representation": "raw",
            "number_of_left_players_agent_controls": args.num_agents,
            "logdir": '../dumps/' + env_name + "_" + args.exp_name,
            "custom_configs": {
                "evaluation": False,
                "shaping_rewards": args.training_shaping_rewards,
            },
        },

        # === Evaluation Settings ===
        "evaluation_num_workers": args.num_eval_workers,
        "custom_eval_function": eval_fn,
        "evaluation_interval": args.eval_interval,  # in terms of training iteration
        "evaluation_num_episodes": args.eval_episodes,
        "evaluation_parallel_to_training": True,
        "evaluation_config": {
            "env_config": {
                "env_name": args.scenario,
                "stacked": False,
                "rewards": "scoring",
                "write_goal_dumps": False,
                "write_full_episode_dumps": True,
                "render": False,
                "write_video": False,
                "dump_frequency": 10,
                "representation": "raw",
                "number_of_left_players_agent_controls": args.num_agents,
                "logdir": '../eval_dumps/' + env_name + "_" + args.exp_name,
                "custom_configs": {
                    "evaluation": True,
                    "shaping_rewards": None,
                },
            },
            "explore": False,
            "num_cpus_per_worker": 1,
            "num_gpus_per_worker": 0,
        },

        "gamma": args.gamma,
        "lambda": args.gae_lambda,
        "kl_coeff": args.kl_coeff,
        "rollout_fragment_length": args.rollout_fragment_length,
        "train_batch_size": args.train_batch_size,
        "sgd_minibatch_size": args.sgd_minibatch_size,
        "num_sgd_iter": args.num_sgd_iter,
        "lr": args.lr,
        "entropy_coeff": args.entropy_coeff,
        "clip_param": args.clip_param,
        "vf_clip_param": args.vf_clip_param,

        "multiagent": {
            "policies": {"shared_policy"},
            "policy_mapping_fn": lambda agent_id, episode, worker, **kwargs: "shared_policy",
            "policies_to_train": ["shared_policy"],
        },
    }

    stop = {"timesteps_total": args.stop_timesteps}

    tune.run(
        PPOTrainer,
        name=env_name + "_" + args.exp_name,
        checkpoint_freq=50,
        keep_checkpoints_num=5,
        checkpoint_at_end=True,
        local_dir="../ray_results",
        restore=args.from_checkpoint,
        stop=stop,
        config=config,
    )

    ray.shutdown()
