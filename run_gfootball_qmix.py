import argparse
from typing import Optional, Dict
from gym.spaces import Tuple

import yaml
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env import BaseEnv
from ray.rllib.utils.typing import PolicyID
from ray.rllib.policy.policy import Policy
from ray.rllib.agents.qmix import QMixTrainer
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode

from envs.gfootball import FootballMultiAgentEnv
from envs.gfootball import eval_fn

EXAMPLE_USAGE = """
python run_shared_ppo_gfootball.py
"""


def create_args():
    parser = argparse.ArgumentParser(
        description="Train gfootball agents with parameter sharing PPO.",
        epilog=EXAMPLE_USAGE)
    parser.add_argument(
        "-f",
        "--config-file",
        default="./configs/gfootball_qmix.yaml",
        type=str,
        help="Use config options from this yaml file.")
    parser.add_argument("--num-agents", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
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
        episode.custom_metrics["score"] = episode.last_info_for("group_1")["_group_info"][0]["score"]


if __name__ == "__main__":
    args = create_args()
    ray.init()

    assert args.env == "football"
    env_name = "gfootball_" + args.scenario

    def env_creator(cfg):
        env = FootballMultiAgentEnv(cfg)
        agent_list = [f"agent_{i}" for i in list(range(env.num_agents))]
        grouping = {
            "group_1": agent_list,
        }
        obs_space = Tuple([env.observation_space for _ in agent_list])
        act_space = Tuple([env.action_space for _ in agent_list])
        return env.with_agent_groups(
            grouping, obs_space=obs_space, act_space=act_space
        )

    register_env(env_name, env_creator)

    config = {
        "framework": "torch",
        "log_level": "INFO",
        "callbacks": CustomMetricsCallback,
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
            "write_full_episode_dumps": False,
            "render": False,
            "write_video": False,
            "dump_frequency": 200,
            "representation": "raw",
            "number_of_left_players_agent_controls": args.num_agents,
            "logdir": './dumps/' + env_name + "_" + args.exp_name,
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
                "dump_frequency": 200,
                "representation": "raw",
                "number_of_left_players_agent_controls": args.num_agents,
                "logdir": './eval_dumps/' + env_name + "_" + args.exp_name,
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
        "rollout_fragment_length": args.rollout_fragment_length,
        "train_batch_size": args.train_batch_size,
        "lr": args.lr,
    }

    stop = {"timesteps_total": args.stop_timesteps}

    tune.run(
        QMixTrainer,
        name=env_name + "_" + args.exp_name,
        checkpoint_freq=50,
        keep_checkpoints_num=5,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        restore=args.from_checkpoint,
        stop=stop,
        config=config,
    )

    ray.shutdown()
