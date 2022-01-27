import argparse
import yaml
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer

from agents.teachers import TEACHER_LIST, CALLBACK_LIST
from envs.mpe import VACLMultiAgentCurriculumMPE, eval_fn

EXAMPLE_USAGE = """
python run_vacl_shared_mpe.py
"""


def create_args():
    parser = argparse.ArgumentParser(
        description="Train MPE agents with VACL.",
        epilog=EXAMPLE_USAGE)
    parser.add_argument(
        "-f",
        "--config-file",
        default="../configs/mpe_vacl_shared.yaml",
        type=str,
        help="Use config options from this yaml file.")
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--scenario", type=str, default=None)
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

    args = parser.parse_args()
    with open(args.config_file) as f:
        configs = yaml.safe_load(f)

    # argparse has higher priority than yaml file
    for k, v in vars(args).items():
        if v is not None:
            configs[k] = v

    return argparse.Namespace(**configs)


if __name__ == "__main__":
    args = create_args()
    ray.init()

    teacher = TEACHER_LIST[args.teacher].options(
        name="teacher", num_cpus=1, num_gpus=0).remote(
        config=dict(**args.teacher_config, **{
            "env": args.env,
            "scenario": args.scenario,
            "num_agents_candidates": args.env_config["num_agent_candidates"],
            "num_workers": args.num_workers,
            "num_envs_per_worker": args.num_envs_per_worker,
        }), seed=args.seed)

    assert args.env == "mpe"
    env_name = args.env + "_" + args.scenario
    register_env(env_name, lambda cfg: VACLMultiAgentCurriculumMPE(cfg))

    config = {
        "framework": "torch",
        "log_level": "INFO",
        "callbacks": CALLBACK_LIST[args.teacher],
        "observation_filter": "NoFilter",
        "seed": args.seed,
        "batch_mode": "complete_episodes",
        "explore": True,

        # === Resource Settings ===
        "num_workers": args.num_workers,
        "num_envs_per_worker": args.num_envs_per_worker,
        "num_cpus_per_worker": 1,
        "num_cpus_for_driver": 1,
        "num_gpus": args.num_gpus,
        "num_gpus_per_worker": 0,

        "env": env_name,
        "env_config": dict(
            **{
                "num_agents": args.env_config["num_agent_candidates"][0],
                "num_observable_agents": args.env_config["num_observable_agents"],
                "episode_length": args.env_config["episode_length"],
            },
            **{
                "seed": args.seed,
                "scenario_name": args.scenario,
                "evaluation": False,
            }),

        # === Evaluation Settings ===
        "evaluation_num_workers": args.num_eval_workers,
        "custom_eval_function": eval_fn,
        "evaluation_interval": args.eval_interval,  # in terms of training iteration
        "evaluation_num_episodes": args.eval_episodes,
        "evaluation_parallel_to_training": True,
        "evaluation_config": {
            "env_config": dict(
                **{
                    "num_agents": args.env_config["num_agent_candidates"][-1],
                    "num_observable_agents": args.env_config["num_observable_agents"],
                    "episode_length": args.env_config["episode_length"],
                },
                **{
                    "seed": args.seed,
                    "scenario_name": args.scenario,
                    "evaluation": True,
                }),
            "explore": False,
            "num_cpus_per_worker": 1,
            "num_gpus_per_worker": 0,
        },

        "gamma": args.gamma,
        "lambda": args.gae_lambda,
        "kl_coeff": args.kl_coeff,
        "rollout_fragment_length": args.env_config["episode_length"],
        "train_batch_size": args.env_config["episode_length"] * args.num_workers * args.num_envs_per_worker,
        "sgd_minibatch_size": args.env_config["episode_length"] * args.num_workers * args.num_envs_per_worker,
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
