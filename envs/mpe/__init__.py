import math
import ray
from ray.rllib.evaluation.metrics import collect_metrics

from .curriculum_hrl_com_env import CurriculumHRLComEnv
from .com_env import ComMPE
from .shared_policy_env import MultiAgentMPE
from .vacl_env import VACLMultiAgentCurriculumMPE, VACLEPMultiAgentCurriculumMPE, VACLComCurriculumMPE
from .curriculum_shared_policy_env import MultiAgentCurriculumMPE
from .curriculum_com_env import CurriculumComMPE


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
    cover_rate = metrics["custom_metrics"]["cover_rate_mean"]
    metrics["cover_rate"] = cover_rate
    print(f"Iter {trainer.iteration} with eval cover rate {cover_rate}.")

    return metrics
