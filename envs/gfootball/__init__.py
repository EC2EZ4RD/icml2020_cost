import math
import ray
from ray.rllib.evaluation.metrics import collect_metrics

from .com_env import FootballMultiAgentComEnv
from .shared_policy_env import FootballMultiAgentEnv

from .curriculum_com_env import FootballCurriculumComEnv
from .curriculum_shared_policy_env import FootballCurriculumMultiAgentEnv
from .curriculum_hrl_shared_policy_env import FootballCurriculumHierarchicalMultiAgentEnv
from .curriculum_hrl_com_env import FootballCurriculumHierarchicalComEnv

from .hrl_shared_policy_env import FootballHierarchicalMultiAgentEnv
from .hrl_com_env import FootballHierarchicalComEnv


def eval_fn(trainer, eval_workers):
    """A custom evaluation function.

    Args:
        trainer (Trainer): trainer class to evaluate.
        eval_workers (WorkerSet): evaluation workers.
    Returns:
        metrics (dict): evaluation metrics dict.
    """

    def _valid_env_config(env):
        print('Evaluation env_config:', env.env_config)
    for w in eval_workers.remote_workers():
        w.foreach_env.remote(_valid_env_config)

    # Evaluation worker set only has local worker.
    if trainer.config["evaluation_num_workers"] == 0:
        for _ in range(trainer.config["evaluation_num_episodes"]):
            eval_workers.local_worker().sample()
    # Evaluation worker set has n remote workers.
    else:
        num_rounds = int(math.ceil(trainer.config["evaluation_num_episodes"] / trainer.config["evaluation_num_workers"]))
        # num_workers = len(eval_workers.remote_workers())
        # num_episodes = num_rounds * num_workers
        for i in range(num_rounds):
            ray.get([w.sample.remote() for w in eval_workers.remote_workers()])

    # Collect the accumulated episodes on the workers, and then summarize the episode stats into a metrics dict.
    metrics = collect_metrics(eval_workers.local_worker(), eval_workers.remote_workers())

    # Put custom values in the metrics dict.
    # NOTE: eval with only scoring reward, so that this is actually win rate.
    win_rate = len(list(filter(lambda x: (x > 0), metrics["hist_stats"]["episode_reward"]))) / len(metrics["hist_stats"]["episode_reward"])
    metrics["win_rate"] = win_rate
    withdraw_rate = len(list(filter(lambda x: (x == 0), metrics["hist_stats"]["episode_reward"]))) / len(metrics["hist_stats"]["episode_reward"])
    metrics["withdraw_rate"] = withdraw_rate
    lose_rate = len(list(filter(lambda x: (x < 0), metrics["hist_stats"]["episode_reward"]))) / len(metrics["hist_stats"]["episode_reward"])
    metrics["lose_rate"] = lose_rate
    print(f"Iter {trainer.iteration} with win rate {win_rate} against bots.")

    return metrics
