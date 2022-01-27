from typing import Dict, List, Type, Union

from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.torch_ops import explained_variance
from ray.rllib.utils.typing import TensorType
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

from .policy import PPOComTorchPolicy

def kl_and_loss_stats(policy: Policy,
                      train_batch: SampleBatch) -> Dict[str, TensorType]:
    """Stats function for PPO. Returns a dict with important KL and loss stats.

    Args:
        policy (Policy): The Policy to generate stats for.
        train_batch (SampleBatch): The SampleBatch (already) used for training.

    Returns:
        Dict[str, TensorType]: The stats dict.
    """
    return {
        "cur_kl_coeff": policy.kl_coeff,
        "cur_lr": policy.cur_lr,
        "total_loss": policy._total_loss,
        "policy_loss": policy._mean_policy_loss,
        "vf_loss": policy._mean_vf_loss,
        "vf_explained_var": policy._vf_explained_var,
        "kl": policy._mean_kl_loss,
        "entropy": policy._mean_entropy,
        "entropy_coeff": policy.entropy_coeff,
        "pred_loss":policy._pred_loss,
    }

def ppo_surrogate_loss_with_variable_agents(
        policy: Policy, model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    """Constructs the loss for Proximal Policy Objective under `complete_episodes` mode.
    Modification of the original `ppo_surrogate_loss` for multi-agent PPO with variable agent number.

    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[ActionDistribution]): The action distr. class.
        train_batch (SampleBatch): The training data.

    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list of loss tensors.
    """
    logits, state = model.from_batch(train_batch, is_training=True)

    pred_act = model.predict_function()
    pred_loss_func = torch.nn.NLLLoss()
    action_dim = pred_act.shape[2]
    batch_pred_loss = 0

    len_vf_pred = len(train_batch[SampleBatch.VF_PREDS][0])
    sgd_batch_size = len(train_batch[SampleBatch.DONES])
    num_agents_batch_size = len(train_batch[SampleBatch.INFOS])
    # split the train_batch into sub-batches with the same num_agents
    split_indices = [0]  # indices of each sub-batch
    if len_vf_pred != 1 and sgd_batch_size != 0 and num_agents_batch_size != 0:  # not dummy init
        # num_agents of each sub-batch
        batch_num_agents = [train_batch[SampleBatch.INFOS][0]["num_agents"]]
        for i, s in enumerate(train_batch[SampleBatch.INFOS]):
            if i > 0 and s["num_agents"] != batch_num_agents[-1]:
                batch_num_agents.append(s["num_agents"])
                split_indices.append(i)
    elif sgd_batch_size == 0 or num_agents_batch_size == 0:
        return torch.randn(1, requires_grad=True)
    else:  # dummy init
        batch_num_agents = [1]
    split_indices.append(len(train_batch[SampleBatch.INFOS]))
    assert len(batch_num_agents) == len(split_indices) - 1

    # if state:
    #     raise NotImplementedError("RNN is not supported since num_agents can change over time")

    # logp, entropy, kl for all possible agents
    curr_action_dist = dist_class(logits, model)
    prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS], model)
    batch_logps = curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
    batch_entropies = curr_action_dist.entropy()
    batch_action_kl = prev_action_dist.kl(curr_action_dist)

    # stats over the whole train_batch
    batch_total_loss = torch.zeros((max(batch_num_agents),)).to(batch_logps.device)
    batch_total_policy_loss = torch.zeros((max(batch_num_agents),)).to(batch_logps.device)
    batch_total_vf_loss = torch.zeros((max(batch_num_agents),)).to(batch_logps.device)
    batch_total_entropy = torch.zeros((max(batch_num_agents),)).to(batch_logps.device)
    batch_total_kl_loss = 0

    # loss for sub-batches different num_agents
    for idx in range(len(batch_num_agents)):
        num_agents = batch_num_agents[idx]
        # index-selection w.r.t. current num_agents
        logps = batch_logps[split_indices[idx]:split_indices[idx + 1], :num_agents]
        entropies = batch_entropies[split_indices[idx]:split_indices[idx + 1], :num_agents]
        action_kl = batch_action_kl[split_indices[idx]:split_indices[idx + 1], :num_agents]
        action_logp = train_batch[SampleBatch.ACTION_LOGP][split_indices[idx]:split_indices[idx + 1], :num_agents]
        vf_preds = train_batch[SampleBatch.VF_PREDS][split_indices[idx]:split_indices[idx + 1], :num_agents]
        advantages = train_batch[Postprocessing.ADVANTAGES][split_indices[idx]:split_indices[idx + 1], :num_agents]
        value_targets = train_batch[Postprocessing.VALUE_TARGETS][split_indices[idx]:split_indices[idx + 1], :num_agents]
        value_fn_out = model.value_function()[split_indices[idx]:split_indices[idx + 1], :num_agents]

        # pred_action_loss w.r.t current num_agents
        batch_pred_loss += pred_loss_func(nn.LogSoftmax(dim=1)(
            pred_act[split_indices[idx]:split_indices[idx + 1], :num_agents].reshape(-1, action_dim)),
            train_batch[SampleBatch.ACTIONS][split_indices[idx]:split_indices[idx + 1], :num_agents].reshape(-1).to(torch.long))

        batch_total_kl_loss += torch.sum(torch.sum(action_kl, axis=1))

        for i in range(num_agents):
            logp_ratio = torch.exp(logps[:, i] - action_logp[:, i])

            batch_total_entropy[i] += torch.sum(entropies[:, i])

            surrogate_loss = torch.min(
                advantages[..., i] * logp_ratio,
                advantages[..., i] * torch.clamp(
                    logp_ratio, 1 - policy.config["clip_param"],
                                1 + policy.config["clip_param"]))
            batch_total_policy_loss[i] += torch.sum(-surrogate_loss)

            # Compute a value function loss.
            if policy.config["use_critic"]:
                prev_value_fn_out = vf_preds[..., i]
                vf_loss1 = torch.pow(value_fn_out[..., i] - value_targets[..., i], 2.0)
                vf_clipped = prev_value_fn_out + torch.clamp(
                    value_fn_out[..., i] - prev_value_fn_out, -policy.config["vf_clip_param"],
                    policy.config["vf_clip_param"])
                vf_loss2 = torch.pow(vf_clipped - value_targets[..., i], 2.0)
                vf_loss = torch.max(vf_loss1, vf_loss2)
                batch_total_vf_loss[i] += torch.sum(vf_loss)
                batch_total_loss[i] += torch.sum(
                    -surrogate_loss + policy.kl_coeff * action_kl[:, i] +
                    policy.config["vf_loss_coeff"] * vf_loss -
                    policy.entropy_coeff * entropies[:, i])
            # Ignore the value function.
            else:
                batch_total_vf_loss[i] = 0.0
                batch_total_loss[i] += torch.sum(-surrogate_loss +
                                                 policy.kl_coeff * action_kl[:, i] -
                                                 policy.entropy_coeff * entropies[:, i])

    # Store stats in policy for stats_fn.
    loss_data = []
    for i in range(max(batch_num_agents)):
        loss_data.append({
            "total_loss": torch.div(batch_total_loss[i], sgd_batch_size),
            "mean_policy_loss": torch.div(batch_total_policy_loss[i], sgd_batch_size),
            "mean_vf_loss": torch.div(batch_total_vf_loss[i], sgd_batch_size),
            "mean_entropy": torch.div(batch_total_entropy[i], sgd_batch_size),
        })

    # Sum the loss of each agent.
    policy._total_loss = torch.sum(torch.stack([o["total_loss"] for o in loss_data]))
    policy._mean_policy_loss = torch.mean(
        torch.stack([o["mean_policy_loss"] for o in loss_data]))
    policy._mean_vf_loss = torch.mean(
        torch.stack([o["mean_vf_loss"] for o in loss_data]))
    policy._mean_entropy = torch.mean(
        torch.stack([o["mean_entropy"] for o in loss_data]))
    policy._vf_explained_var = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS],
        policy.model.value_function())
    policy._mean_kl_loss = torch.div(batch_total_kl_loss, sgd_batch_size)
    policy._pred_loss = batch_pred_loss
    policy._total_loss += 0.1 * policy._pred_loss

    return policy._total_loss


PPOComCurriculumTorchPolicy = PPOComTorchPolicy.with_updates(
    name="PPOComCurriculumTorchPolicy",
    stats_fn=kl_and_loss_stats,
    loss_fn=ppo_surrogate_loss_with_variable_agents,
)
