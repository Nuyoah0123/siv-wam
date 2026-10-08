"""On-policy, sequential grouped rollouts and action-head-only GRPO updates."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch
from .objectives import group_advantages, gaussian_log_prob, gaussian_kl, clipped_surrogate


class GRPOTrainer:
    def __init__(self, actor, environment, config, output_dir, identity=None):
        self.actor, self.env, self.config = actor, environment, config
        self.head = actor.head
        self.reference = deepcopy(self.head).eval().requires_grad_(False)
        self.optimizer = torch.optim.AdamW(self.head.parameters(), lr=config.learning_rate, weight_decay=0)
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.identity = identity or {'backend': type(actor).__name__}
        self.update = 0
        self._frozen_versions = {name: p._version for name, p in actor.model.named_parameters() if not p.requires_grad}
        self._check_trainable_scope()

    @property
    def device(self):
        return next(self.head.parameters()).device

    def _check_trainable_scope(self):
        expected = {id(p) for p in self.head.parameters()}
        actual = {id(p) for p in self.actor.model.parameters() if p.requires_grad}
        if actual != expected:
            raise RuntimeError('Only action_proj_out parameters may be trainable in Stage II')
        for name, p in self.actor.model.named_parameters():
            if name in self._frozen_versions and (p._version != self._frozen_versions[name] or p.grad is not None):
                raise RuntimeError(f'Frozen parameter changed: {name}')

    def _episode(self, scene_seed, sample_seed, deterministic=False):
        observation = self.env.reset(scene_seed)
        self.actor.reset(observation)
        generator = torch.Generator(device=self.device).manual_seed(sample_seed)
        transitions, rewards = [], []
        dense_total = sparse_total = 0.0
        steps = 0
        success = False
        for chunk_id in range(self.config.max_chunks):
            chunk = self.actor.sample(observation, generator, scene_seed + chunk_id, deterministic=deterministic)
            result = self.env.step_chunk(chunk)
            if result.executed_steps < 1:
                raise RuntimeError('Environment executed no actions; do not optimize an invalid rollout')
            reward = self.config.dense_weight * result.dense + self.config.sparse_weight * result.sparse
            if not torch.isfinite(torch.tensor(reward)):
                raise FloatingPointError('Non-finite rollout reward')
            rewards.append(reward)
            dense_total += result.dense
            sparse_total += result.sparse
            steps += result.executed_steps
            transitions.extend(chunk.transitions)
            observation, success = result.observation, result.success
            if result.done:
                break
        total_return = sum(self.config.discount ** i * r for i, r in enumerate(rewards))
        return transitions, {'return': total_return, 'dense': dense_total, 'sparse': sparse_total,
                             'success': bool(success), 'executed_steps': steps,
                             'chunks': len(rewards), 'scene_seed': scene_seed, 'sample_seed': sample_seed}

    def _distribution(self, record, head):
        features = record.features.to(self.device)
        return record.state.to(self.device) + record.dt * head(features)

    def optimize(self, episodes, returns):
        advantages = group_advantages(returns).to(self.device)
        if len(episodes) != len(returns) or any(not records for records in episodes):
            raise ValueError('Each candidate needs an on-policy denoising trace')
        # Verify old log probabilities before the first optimizer step; catches stale buffers.
        with torch.no_grad():
            for records in episodes:
                for record in records:
                    lp = gaussian_log_prob(record.next_state.to(self.device), self._distribution(record, self.head),
                                           record.std, record.active.to(self.device))
                    if not torch.allclose(lp, record.old_log_prob.to(self.device), atol=2e-3, rtol=1e-6):
                        raise RuntimeError('Rollout log probabilities do not match current behavior policy')
        info = {'policy_loss': 0.0, 'reference_kl': 0.0, 'old_policy_kl': 0.0,
                'clip_fraction': 0.0, 'epochs': 0, 'zero_advantage_group': bool((advantages == 0).all())}
        for epoch in range(self.config.epochs):
            self.optimizer.zero_grad(set_to_none=True)
            sums = {key: 0.0 for key in ('policy_loss', 'reference_kl', 'old_policy_kl', 'clip_fraction')}
            for advantage, records in zip(advantages, episodes):
                weight = 1.0 / (len(episodes) * len(records))
                for record in records:
                    active = record.active.to(self.device)
                    mean = self._distribution(record, self.head)
                    new_lp = gaussian_log_prob(record.next_state.to(self.device), mean, record.std, active)
                    policy_loss, ratio = clipped_surrogate(new_lp, record.old_log_prob.to(self.device),
                                                           advantage, self.config.clip_epsilon)
                    with torch.no_grad():
                        reference_mean = self._distribution(record, self.reference)
                    ref_kl = gaussian_kl(mean, reference_mean, record.std, active)
                    old_kl = gaussian_kl(mean, record.old_mean.to(self.device), record.std, active)
                    loss = policy_loss + self.config.kl_beta * ref_kl
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite GRPO objective')
                    (weight * loss).backward()
                    sums['policy_loss'] += weight * float(policy_loss.detach())
                    sums['reference_kl'] += weight * float(ref_kl.detach())
                    sums['old_policy_kl'] += weight * float(old_kl.detach())
                    sums['clip_fraction'] += weight * float((ratio - 1).abs() > self.config.clip_epsilon)
            # Stop before an additional update if the behavior policy is already too far away.
            if sums['old_policy_kl'] > self.config.target_kl:
                self.optimizer.zero_grad(set_to_none=True)
                break
            norm = torch.nn.utils.clip_grad_norm_(self.head.parameters(), self.config.grad_clip, error_if_nonfinite=True)
            self.optimizer.step()
            self._check_trainable_scope()
            info.update(sums, epochs=epoch + 1, grad_norm=float(norm))
        return info

    def train(self):
        history = []
        while self.update < self.config.updates:
            scene_seed = self.config.seed + self.update
            records, summaries = [], []
            # No parameter updates until ALL candidates from this reset seed are collected.
            for member in range(self.config.group_size):
                sample_seed = self.config.seed + 1_000_003 * (self.update + 1) + member
                trace, summary = self._episode(scene_seed, sample_seed)
                records.append(trace)
                summaries.append(summary)
            stats = self.optimize(records, [x['return'] for x in summaries])
            self.update += 1
            stats.update(update=self.update, mean_return=sum(x['return'] for x in summaries) / len(summaries),
                         success_rate=sum(x['success'] for x in summaries) / len(summaries),
                         candidates=summaries, config_hash=self.config_hash())
            with (self.output_dir / 'train.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(stats, ensure_ascii=False) + '\n')
            print(json.dumps({k: v for k, v in stats.items() if k != 'candidates'}), flush=True)
            history.append(stats)
            if self.update % self.config.save_every == 0 or self.update == self.config.updates:
                self.save(self.output_dir / 'latest.pt')
        return history

    def evaluate(self, episodes=10, deterministic=False):
        if episodes < 1:
            raise ValueError('episodes must be positive')
        results = [self._episode(self.config.seed + 100_000 + i, self.config.seed + 200_000 + i,
                                 deterministic=deterministic)[1] for i in range(episodes)]
        summary = {'episodes': episodes, 'deterministic': deterministic,
                   'success_rate': sum(x['success'] for x in results) / episodes,
                   'mean_return': sum(x['return'] for x in results) / episodes,
                   'results': results, 'identity': self.identity}
        (self.output_dir / 'evaluation.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        return summary

    def config_hash(self):
        return hashlib.sha256(json.dumps({'config': asdict(self.config), 'identity': self.identity},
                                         sort_keys=True).encode()).hexdigest()

    def save(self, path):
        path = Path(path)
        state = {'format_version': 1, 'update': self.update, 'config': asdict(self.config),
                 'identity': self.identity, 'config_hash': self.config_hash(),
                 'action_head': {k: v.detach().cpu() for k, v in self.head.state_dict().items()},
                 'reference_head': {k: v.detach().cpu() for k, v in self.reference.state_dict().items()},
                 'optimizer': self.optimizer.state_dict()}
        temporary = path.with_suffix(path.suffix + '.tmp')
        torch.save(state, temporary)
        temporary.replace(path)

    def load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=True)
        if state.get('format_version') != 1 or state['identity'] != self.identity:
            raise ValueError('Checkpoint format or model/environment identity does not match')
        for key, value in asdict(self.config).items():
            if key not in {'updates', 'save_every'} and state['config'][key] != value:
                raise ValueError(f'Resume configuration mismatch: {key}')
        self.head.load_state_dict(state['action_head'], strict=True)
        self.reference.load_state_dict(state['reference_head'], strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.update = int(state['update'])
        self._check_trainable_scope()
