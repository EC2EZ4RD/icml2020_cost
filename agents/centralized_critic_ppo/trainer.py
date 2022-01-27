from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG, validate_config, execution_plan

from .policy import CCPPOTorchPolicy


CCPPOTrainer = build_trainer(
    name="CCPPOTrainer",
    default_config=DEFAULT_CONFIG,
    validate_config=validate_config,
    default_policy=CCPPOTorchPolicy,
    execution_plan=execution_plan,
)
