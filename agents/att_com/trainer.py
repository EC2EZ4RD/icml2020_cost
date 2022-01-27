from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG, validate_config, execution_plan

from .policy import PPOComTorchPolicy
from .curriculum_policy import PPOComCurriculumTorchPolicy


PPOComTrainer = build_trainer(
    name="PPOComTrainer",
    default_config=DEFAULT_CONFIG,
    validate_config=validate_config,
    default_policy=PPOComTorchPolicy,
    execution_plan=execution_plan,
)


PPOComCurriculumTrainer = build_trainer(
    name="PPOComCurriculumTrainer",
    default_config=DEFAULT_CONFIG,
    validate_config=validate_config,
    default_policy=PPOComCurriculumTorchPolicy,
    execution_plan=execution_plan,
)
