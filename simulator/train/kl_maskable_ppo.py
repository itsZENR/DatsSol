"""
MaskablePPO with KL-regularization toward a frozen reference policy (e.g. BC).

Adds `β * KL(current || ref)` to the loss, preventing drift from BC-optimum.
"""
import torch
import torch.nn.functional as F
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


class KLMaskablePPO(MaskablePPO):
    """MaskablePPO with KL-regularization toward a reference policy.

    Usage:
        ref_model = MaskablePPO.load("ppo_bc_attn")
        model = KLMaskablePPO("MultiInputPolicy", env, ...)
        model.ref_policy = ref_model.policy
        model.kl_coef = 0.2
        model.learn(...)
    """
    ref_policy: MaskableActorCriticPolicy | None = None
    kl_coef: float = 0.0

    def train(self) -> None:
        if self.ref_policy is None or self.kl_coef == 0.0:
            return super().train()

        # Copy of super().train() with added KL-to-reference loss.
        # This is a minimal patch — most of train is reused via super's logic.
        # For simplicity, we apply KL penalty as an additional term accumulated
        # across rollouts.
        import time
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, torch.nn.Module):  # no
                    pass
                # Standard PPO forward
                if hasattr(rollout_data, 'action_masks'):
                    values, log_prob, entropy = self.policy.evaluate_actions(
                        rollout_data.observations, actions.long(),
                        action_masks=rollout_data.action_masks,
                    )
                else:
                    values, log_prob, entropy = self.policy.evaluate_actions(
                        rollout_data.observations, actions.long(),
                    )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio, 1 - self.clip_range, 1 + self.clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                value_loss = F.mse_loss(rollout_data.returns, values)

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                # KL-regularization: current policy vs reference (BC)
                kl_loss = torch.tensor(0.0, device=self.device)
                if self.kl_coef > 0:
                    with torch.no_grad():
                        ref_distr = self.ref_policy.get_distribution(
                            rollout_data.observations
                        )
                    cur_distr = self.policy.get_distribution(rollout_data.observations)
                    # For MultiDiscrete distributions, use `kl_divergence` over each sub-dist
                    try:
                        kl_per = torch.distributions.kl_divergence(
                            cur_distr.distribution, ref_distr.distribution
                        )
                        kl_loss = kl_per.sum(dim=-1).mean()
                    except Exception:
                        # Fallback: mean log-prob difference
                        with torch.no_grad():
                            ref_log_prob = self.ref_policy.evaluate_actions(
                                rollout_data.observations, actions.long()
                            )[1]
                        kl_loss = (log_prob - ref_log_prob).mean()

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.kl_coef * kl_loss
                )

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().item()
                    approx_kl_divs.append(approx_kl)

            self._n_updates += 1

        self.logger.record("train/kl_to_ref", float(kl_loss.item()))
