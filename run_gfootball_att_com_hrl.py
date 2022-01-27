import argparse
from typing import Optional, Dict
import yaml
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env import BaseEnv
from ray.rllib.utils.typing import PolicyID
from ray.rllib.policy.policy import Policy, PolicySpec
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.models import ModelCatalog

from agents.hrl import HRLTrainer
from agents.att_com import PPOComTorchPolicy
from envs.gfootball import FootballHierarchicalComEnv, eval_fn
from models.action_dist import TorchHomogeneousMultiActionDistribution
from models import AttComModel

EXAMPLE_USAGE = """
python run_hrl_att_com_gfootball.py
"""


def create_args():
    parser = argparse.ArgumentParser(
        description="Train gfootball agents with HRL. High and low levels both use parameter sharing PPO.",
        epilog=EXAMPLE_USAGE)
    parser.add_argument(
        "-f",
        "--config-file",
        default="./configs/gfootball_att_com_hrl.yaml",
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
        episode.custom_metrics["score"] = episode.last_info_for("high_level_policy")["score"]


if __name__ == "__main__":
    args = create_args()
    ray.init()

    ModelCatalog.register_custom_model("att_com_model", AttComModel)
    ModelCatalog.register_custom_action_dist(
        "hom_multi_action", TorchHomogeneousMultiActionDistribution
    )

    assert args.env == "football"
    env_name = "gfootball_" + args.scenario
    register_env(env_name, lambda cfg: FootballHierarchicalComEnv(cfg))

    env_config = {
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
            "max_num_agents": args.num_agents,
            "context_size": args.context_size,
            "context_type": args.context_type,
            "high_level_interval": args.high_level_interval,
        },
    }

    temp_env = FootballHierarchicalComEnv(env_config)

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id.startswith("agent_"):
            return "low_level_policy"
        else:
            return "high_level_policy"

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
        "env_config": env_config,

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
                "write_full_episode_dumps": False,
                "render": False,
                "write_video": False,
                "dump_frequency": 10,
                "representation": "raw",
                "number_of_left_players_agent_controls": args.num_agents,
                "logdir": './eval_dumps/' + env_name + "_" + args.exp_name,
                "custom_configs": {
                    "evaluation": True,
                    "shaping_rewards": None,
                    "max_num_agents": args.num_agents,
                    "context_size": args.context_size,
                    "context_type": args.context_type,
                    "high_level_interval": args.high_level_interval,
                },
            },
            "explore": False,
            "num_cpus_per_worker": 1,
            "num_gpus_per_worker": 0,
        },
        
        # === Hierarchical Training ===
        "multiagent": {
            "policies": {
                "high_level_policy": PolicySpec(
                    policy_class=PPOComTorchPolicy,
                    observation_space=temp_env.high_level_observation_space,
                    action_space=temp_env.high_level_action_space,
                    config={
                        "model": {
                            "custom_model": "att_com_model",
                            "custom_action_dist": "hom_multi_action",
                            "custom_model_config": args.high_level_config["model_config"],
                        },
                        "gamma": args.high_level_config["gamma"],
                        "lambda": args.high_level_config["gae_lambda"],
                        "kl_coeff": args.high_level_config["kl_coeff"],
                        "rollout_fragment_length": args.high_level_config["rollout_fragment_length"],
                        "train_batch_size": args.high_level_config["train_batch_size"],
                        "sgd_minibatch_size": args.high_level_config["sgd_minibatch_size"],
                        "num_sgd_iter": args.high_level_config["num_sgd_iter"],
                        "lr": args.high_level_config["lr"],
                        "entropy_coeff": args.high_level_config["entropy_coeff"],
                        "clip_param": args.high_level_config["clip_param"],
                        "vf_clip_param": args.high_level_config["vf_clip_param"],
                    }
                ),
                "low_level_policy": PolicySpec(
                    policy_class=PPOTorchPolicy,
                    observation_space=temp_env.low_level_observation_space,
                    action_space=temp_env.low_level_action_space,
                    config={
                        "gamma": args.low_level_config["gamma"],
                        "lambda": args.low_level_config["gae_lambda"],
                        "kl_coeff": args.low_level_config["kl_coeff"],
                        "rollout_fragment_length": args.low_level_config["rollout_fragment_length"],
                        "train_batch_size": args.low_level_config["train_batch_size"],
                        "sgd_minibatch_size": args.low_level_config["sgd_minibatch_size"],
                        "num_sgd_iter": args.low_level_config["num_sgd_iter"],
                        "lr": args.low_level_config["lr"],
                        "entropy_coeff": args.low_level_config["entropy_coeff"],
                        "clip_param": args.low_level_config["clip_param"],
                        "vf_clip_param": args.low_level_config["vf_clip_param"],
                    }
                ),
            },
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": ["high_level_policy", "low_level_policy"],
        },
    }

    stop = {"timesteps_total": args.stop_timesteps}

    tune.run(
        HRLTrainer,
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
