from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.utils.typing import TrainerConfigDict
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.common import _get_shared_metrics
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches, StandardizeFields, SelectExperiences
from ray.rllib.execution.concurrency_ops import Concurrently
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.train_ops import TrainOneStep
from ray.util.iter import LocalIterator
from ray.rllib.agents.ppo.ppo import UpdateKL


def hrl_training_flow(trainer: Trainer, workers: WorkerSet,
                      config: TrainerConfigDict) -> LocalIterator[dict]:
    """Execution plan of the HRL algorithm. Defines the distributed dataflow.
    Args:
        workers (WorkerSet): The WorkerSet for training the Polic(y/ies)
            of the Trainer.
        config (TrainerConfigDict): The trainer's configuration dict.
    Returns:
        LocalIterator[dict]: A local iterator over training metrics.
    """

    def add_high_level_metrics(batch):
        print("High-level policy learning on samples from",
              batch.policy_batches.keys(), "env steps", batch.env_steps(),
              "agent steps", batch.agent_steps())
        metrics = _get_shared_metrics()
        metrics.counters["num_high_level_steps"] += batch.agent_steps()
        return batch

    def add_low_level_metrics(batch):
        print("Low-level policy learning on samples from",
              batch.policy_batches.keys(), "env steps", batch.env_steps(),
              "agent steps", batch.agent_steps())
        metrics = _get_shared_metrics()
        metrics.counters["num_low_level_steps_of_all_agents"] += batch.agent_steps()
        return batch

    # Generate common experiences.
    rollouts = ParallelRollouts(workers, mode="bulk_sync")
    high_level_rollouts, low_level_rollouts = rollouts.duplicate(n=2)

    # High-level PPO sub-flow.
    high_level_train_op = high_level_rollouts.for_each(SelectExperiences(["high_level_policy"])) \
        .combine(ConcatBatches(
        min_batch_size=config["multiagent"]["policies"]["high_level_policy"].config["train_batch_size"],
        count_steps_by="env_steps")) \
        .for_each(add_high_level_metrics) \
        .for_each(StandardizeFields(["advantages"])) \
        .for_each(TrainOneStep(
        workers,
        policies=["high_level_policy"],
        num_sgd_iter=config["multiagent"]["policies"]["high_level_policy"].config["num_sgd_iter"],
        sgd_minibatch_size=config["multiagent"]["policies"]["high_level_policy"].config["sgd_minibatch_size"])) \
        .for_each(lambda t: t[1]) \
        .for_each(UpdateKL(workers))

    # Low-level PPO sub-flow.
    low_level_train_op = low_level_rollouts.for_each(SelectExperiences(["low_level_policy"])) \
        .combine(ConcatBatches(
        min_batch_size=config["multiagent"]["policies"]["low_level_policy"].config["train_batch_size"],
        count_steps_by="env_steps")) \
        .for_each(add_low_level_metrics) \
        .for_each(StandardizeFields(["advantages"])) \
        .for_each(TrainOneStep(
        workers,
        policies=["low_level_policy"],
        num_sgd_iter=config["multiagent"]["policies"]["low_level_policy"].config["num_sgd_iter"],
        sgd_minibatch_size=config["multiagent"]["policies"]["low_level_policy"].config["sgd_minibatch_size"])) \
        .for_each(lambda t: t[1]) \
        .for_each(UpdateKL(workers))

    # Combined training flow
    train_op = Concurrently([low_level_train_op, high_level_train_op], mode="async", output_indexes=[1])

    # Warn about bad reward scales and return training metrics.
    return StandardMetricsReporting(train_op, workers, config)


HRLTrainer = build_trainer(
    name="HRLTrainer",
    default_policy=None,
    execution_plan=hrl_training_flow,
)
