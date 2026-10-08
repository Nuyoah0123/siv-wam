"""Optional tests with an actual tiny instance of the repository's Transformer."""
import importlib.util
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import torch
from siv_wam.rl.config import RLConfig
from siv_wam.rl.trainer import GRPOTrainer
from siv_wam.rl.trajectory import sample_transition

HAS_MODEL_DEPS = all(importlib.util.find_spec(p) is not None for p in ('diffusers', 'transformers', 'einops', 'safetensors'))


@unittest.skipUnless(HAS_MODEL_DEPS, 'Install model dependencies to exercise the actual Transformer')
class ActualTransformerTests(unittest.TestCase):
    def setUp(self):
        from siv_wam.modules.model import WanTransformer3DModel
        torch.manual_seed(10)
        self.model = WanTransformer3DModel(patch_size=[1, 1, 1], num_attention_heads=2,
            attention_head_dim=12, in_channels=4, out_channels=4, action_dim=2,
            text_dim=8, freq_dim=8, ffn_dim=48, num_layers=1, attn_mode='torch')
        self.model.eval().requires_grad_(False)
        self.model.action_proj_out.requires_grad_(True)
        self.inputs = {'noisy_latents': torch.randn(1, 2, 1, 2, 1),
                       'text_emb': torch.randn(1, 2, 8),
                       'grid_id': torch.zeros(1, 4, 2), 'timesteps': torch.tensor([[500.]])}

    def test_feature_interface_preserves_existing_forward_output(self):
        with torch.no_grad():
            expected = self.model(self.inputs, action_mode=True)
            features = self.model(self.inputs, action_mode=True, return_action_features=True)
            actual = self.model.action_proj_out(features)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_two_stream_train_interface_remains_supported(self):
        """Base video/action training may omit the optional SIV stream."""
        from siv_wam.modules.model import FlexAttnFunc, WanTransformerBlock
        model = self.model.to(dtype=torch.bfloat16)
        latent = torch.randn(1, 4, 1, 2, 1, dtype=torch.bfloat16)
        action = torch.randn(1, 2, 1, 2, 1, dtype=torch.bfloat16)
        stream = {
            'noisy_latents': latent.clone(), 'latent': latent.clone(),
            'timesteps': torch.tensor([[500.]], dtype=torch.bfloat16),
            'cond_timesteps': torch.zeros(1, 1, dtype=torch.bfloat16),
            'grid_id': torch.zeros(1, 4, 2),
        }
        action_stream = {
            'noisy_latents': action.clone(), 'latent': action.clone(),
            'timesteps': torch.tensor([[500.]], dtype=torch.bfloat16),
            'cond_timesteps': torch.zeros(1, 1, dtype=torch.bfloat16),
            'grid_id': torch.zeros(1, 4, 2),
        }
        inputs = {
            'latent_dict': dict(stream, text_emb=torch.randn(1, 2, 8, dtype=torch.bfloat16)),
            'action_dict': action_stream,
            'chunk_size': 1,
            'window_size': 4,
        }
        def identity_block(self, hidden_states, text_hidden_states, timestep_proj,
                           rotary_emb, update_cache=0, cache_name='pos'):
            return hidden_states

        # The interface regression test does not need to compile FlexAttention;
        # the real attention path is covered by the existing action-mode tests.
        with patch.object(FlexAttnFunc, 'init_mask'), \
                patch.object(WanTransformerBlock, 'forward', identity_block), \
                torch.no_grad():
            video_out, action_out = model(inputs, train_mode=True)
        self.assertEqual(tuple(video_out.shape), (1, 2, 4))
        self.assertEqual(tuple(action_out.shape), (1, 2, 2))

    def test_grpo_updates_real_action_head_without_changing_video_or_siv(self):
        video = dict(self.inputs, noisy_latents=torch.randn(1, 4, 1, 2, 1),
                     noisy_mask_latents=torch.randn(1, 4, 1, 2, 1))
        with torch.no_grad():
            before_video, before_siv = self.model(video)
            features = self.model(self.inputs, action_mode=True, return_action_features=True).flatten(0, 1)
            before_action = self.model(self.inputs, action_mode=True)
        state = self.inputs['noisy_latents'].permute(0, 2, 3, 4, 1).reshape(2, 2)
        records = [sample_transition(self.model.action_proj_out, features, state, torch.ones_like(state, dtype=torch.bool),
                                    -.1, .2, torch.Generator().manual_seed(seed))[1] for seed in (1, 2)]
        with tempfile.TemporaryDirectory() as output:
            trainer = GRPOTrainer(SimpleNamespace(model=self.model, head=self.model.action_proj_out),
                                  None, RLConfig(epochs=1, learning_rate=.01), output)
            trainer.optimize([[records[0]], [records[1]]], [0., 1.])
        with torch.no_grad():
            after_video, after_siv = self.model(video)
            after_action = self.model(self.inputs, action_mode=True)
        torch.testing.assert_close(before_video, after_video, rtol=0, atol=0)
        torch.testing.assert_close(before_siv, after_siv, rtol=0, atol=0)
        self.assertFalse(torch.equal(before_action, after_action))


if __name__ == '__main__':
    unittest.main()
