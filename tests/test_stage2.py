import contextlib
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from torch import nn
from siv_wam.rl.config import RLConfig
from siv_wam.rl.geometry import project_world, sample_value_map, projected_reward
from siv_wam.rl.objectives import group_advantages, gaussian_log_prob, gaussian_kl, clipped_surrogate
from siv_wam.rl.robotwin import decode_ee_action
from siv_wam.rl.smoke import SmokeActor, SmokeEnvironment
from siv_wam.rl.trainer import GRPOTrainer
from siv_wam.rl.trajectory import sample_transition

torch.set_num_threads(1)


class GeometryTests(unittest.TestCase):
    def test_projection_and_invalid_targets(self):
        K = torch.tensor([[10., 0, 5], [0, 10, 5], [0, 0, 1]])
        points = torch.tensor([[0., 0, 2], [0, 0, -1], [20, 0, 1], [float('nan'), 0, 1]])
        pixels, valid = project_world(points, K, torch.eye(4), (11, 11))
        self.assertEqual(valid.tolist(), [True, False, False, False])
        heatmap = torch.zeros(11, 11)
        heatmap[5, 5] = 1
        torch.testing.assert_close(sample_value_map(heatmap, pixels, valid), torch.tensor([1., 0, 0, 0]))

    def test_bilinear_interpolation(self):
        heatmap = torch.tensor([[0., 1], [1, 0]])
        response = sample_value_map(heatmap, torch.tensor([[0.5, 0.5]]), torch.tensor([True]))
        self.assertAlmostEqual(response.item(), 0.5)

    def test_world_to_camera_translation(self):
        T = torch.eye(4)
        T[0, 3] = -2
        reward = projected_reward(torch.ones(5, 5), [[2, 0, 1]], torch.eye(3), T)
        self.assertEqual(reward.item(), 1)

    def test_action_frame_and_quaternion_conversion(self):
        action = np.array([.1, .2, .3, 0, 0, 0, 1, 1] * 2)
        initial = np.array([1, 2, 3, 1, 0, 0, 0, 0] * 2)
        output = decode_ee_action(action, initial, 'xyzw', 'wxyz')
        np.testing.assert_allclose(output[:3], [1.1, 2.2, 3.3])
        np.testing.assert_allclose(output[3:7], [1, 0, 0, 0])
        action[3:7] = 0
        with self.assertRaises(ValueError):
            decode_ee_action(action, initial, 'xyzw', 'wxyz')


class ObjectiveTests(unittest.TestCase):
    def test_group_relative_advantages(self):
        torch.testing.assert_close(group_advantages([1, 2, 3]), torch.tensor([-1.2247449, 0, 1.2247449]))
        torch.testing.assert_close(group_advantages([3, 3]), torch.zeros(2))
        with self.assertRaises(ValueError):
            group_advantages([1])

    def test_density_uses_only_active_coordinates(self):
        value = torch.tensor([[0.2, 1000.], [-0.3, 900.]])
        mean = torch.zeros_like(value)
        mask = torch.tensor([[True, False], [True, False]])
        expected = torch.distributions.Normal(0., .5).log_prob(value[:, 0]).sum()
        torch.testing.assert_close(gaussian_log_prob(value, mean, .5, mask), expected)
        self.assertEqual(gaussian_kl(mean, mean, .5, mask).item(), 0)

    def test_clipping_handles_both_advantage_signs(self):
        new = torch.tensor([np.log(1.5), np.log(.5)], dtype=torch.float32, requires_grad=True)
        loss, _ = clipped_surrogate(new, torch.zeros(2), torch.tensor([1., -1.]), .2)
        self.assertAlmostEqual(loss.item(), -.2, places=6)
        loss.backward()
        torch.testing.assert_close(new.grad, torch.zeros(2))

    def test_sampling_preserves_conditioned_and_padding_coordinates(self):
        head = nn.Linear(3, 2)
        features = torch.ones(4, 3)
        state = torch.arange(8.).reshape(4, 2)
        active = torch.tensor([[False, False], [True, False], [True, False], [False, False]])
        nxt, record = sample_transition(head, features, state, active, -.1, .2, torch.Generator().manual_seed(1))
        torch.testing.assert_close(nxt[~active], state[~active])
        self.assertFalse(record.old_log_prob.requires_grad)
        self.assertFalse(record.features.requires_grad)

    def test_invalid_configuration_is_rejected(self):
        for settings in ({'group_size': 1}, {'exploration_std': 0}, {'learning_rate': float('nan')}, {'discount': 2}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                RLConfig(**settings)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = RLConfig(seed=7, updates=2, group_size=8, max_chunks=1,
                               learning_rate=.03, epochs=2, exploration_std=.35, target_kl=.5, save_every=1)

    def trainer(self, name, config=None):
        cfg = config or self.config
        return GRPOTrainer(SmokeActor(cfg), SmokeEnvironment(), cfg, Path(self.tmp.name) / name)

    def test_actual_head_update_with_frozen_world_and_siv(self):
        trainer = self.trainer('freeze')
        before = {k: p.clone() for k, p in trainer.actor.model.state_dict().items()}
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.train()
        after = trainer.actor.model.state_dict()
        self.assertTrue(any(not torch.equal(before[k], after[k]) for k in before if k.startswith('action_proj_out')))
        for name in ('world.weight', 'mask_proj_out.weight'):
            torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for name, p in trainer.actor.model.named_parameters() if not name.startswith('action_proj_out')))

    def test_checkpoint_resume_matches_uninterrupted_training(self):
        full = self.trainer('full')
        split = self.trainer('split', replace(self.config, updates=1))
        with contextlib.redirect_stdout(io.StringIO()):
            full.train()
            split.train()
            resumed = self.trainer('resumed')
            resumed.load(split.output_dir / 'latest.pt')
            resumed.train()
        for name, tensor in full.head.state_dict().items():
            torch.testing.assert_close(tensor, resumed.head.state_dict()[name], rtol=0, atol=0)
        for name, tensor in full.reference.state_dict().items():
            torch.testing.assert_close(tensor, resumed.reference.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(resumed.update, 2)

    def test_stale_rollouts_are_rejected(self):
        trainer = self.trainer('stale')
        trace, result = trainer._episode(1, 2)
        with torch.no_grad():
            trainer.head.bias.add_(1)
        with self.assertRaises(RuntimeError):
            trainer.optimize([trace, trace], [result['return'], result['return']])

    def test_group_reuses_reset_seed_but_not_action_noise(self):
        trainer = self.trainer('groups', replace(self.config, updates=1))
        with contextlib.redirect_stdout(io.StringIO()):
            log = trainer.train()[0]
        candidates = log['candidates']
        self.assertEqual(len({c['scene_seed'] for c in candidates}), 1)
        self.assertEqual(len({c['sample_seed'] for c in candidates}), self.config.group_size)

    def test_zero_reward_group_has_no_policy_update(self):
        trainer = self.trainer('zero')
        records = [trainer._episode(1, seed)[0] for seed in (2, 3)]
        before = {k: p.clone() for k, p in trainer.head.state_dict().items()}
        stats = trainer.optimize(records, [0., 0.])
        self.assertTrue(stats['zero_advantage_group'])
        for key, value in before.items():
            torch.testing.assert_close(value, trainer.head.state_dict()[key], rtol=0, atol=0)

    def test_smoke_training_improves_heldout_return(self):
        trainer = self.trainer('learn', replace(self.config, updates=15, group_size=16))
        before = trainer.evaluate(20, deterministic=True)['mean_return']
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.train()
        after = trainer.evaluate(20, deterministic=True)['mean_return']
        self.assertGreater(after, before + .2)


if __name__ == '__main__':
    unittest.main()
