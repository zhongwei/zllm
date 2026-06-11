"""M3-C: YaRN RoPE Scaling 测试。

YaRN 通过频率缩放实现长序列外推：
- 仅当 end > original_max_position_embeddings 时生效
- 线性 ramp 混合原始/缩放频率
"""

import torch
import pytest

from zllm.model.rope import precompute_freqs_cis


class TestYaRNScaling:
    def _yarn_config(self, **overrides):
        defaults = {
            "original_max_position_embeddings": 128,
            "factor": 4.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "attention_factor": 1.0,
            "type": "yarn",
        }
        defaults.update(overrides)
        return defaults

    def test_yarn_scales_frequencies(self):
        cfg = self._yarn_config()
        cos_base, _ = precompute_freqs_cis(dim=16, end=256, rope_base=1e4)
        cos_yarn, _ = precompute_freqs_cis(dim=16, end=256, rope_base=1e4, rope_scaling=cfg)
        assert not torch.allclose(cos_base.float(), cos_yarn.float(), atol=1e-5)

    def test_yarn_no_effect_below_orig_max(self):
        cfg = self._yarn_config(original_max_position_embeddings=512)
        cos_base, sin_base = precompute_freqs_cis(dim=16, end=256, rope_base=1e4)
        cos_yarn, sin_yarn = precompute_freqs_cis(dim=16, end=256, rope_base=1e4, rope_scaling=cfg)
        assert torch.allclose(cos_base.float(), cos_yarn.float(), atol=1e-5)
        assert torch.allclose(sin_base.float(), sin_yarn.float(), atol=1e-5)

    def test_yarn_attention_factor(self):
        cfg = self._yarn_config(attention_factor=0.5)
        cos1, _ = precompute_freqs_cis(dim=16, end=256, rope_base=1e4, rope_scaling=cfg)
        cfg2 = self._yarn_config(attention_factor=1.0)
        cos2, _ = precompute_freqs_cis(dim=16, end=256, rope_base=1e4, rope_scaling=cfg2)
        assert not torch.allclose(cos1.float(), cos2.float(), atol=1e-3)

    def test_yarn_preserves_shapes(self):
        cfg = self._yarn_config()
        cos, sin = precompute_freqs_cis(dim=16, end=512, rope_base=1e4, rope_scaling=cfg)
        assert cos.shape == (512, 16)
        assert sin.shape == (512, 16)

    def test_yarn_position_0_cos_unchanged(self):
        cfg = self._yarn_config()
        cos, _ = precompute_freqs_cis(dim=16, end=256, rope_base=1e4, rope_scaling=cfg)
        assert torch.allclose(cos[0].float(), torch.ones(16), atol=1e-5)

    def test_yarn_position_0_sin_unchanged(self):
        cfg = self._yarn_config()
        _, sin = precompute_freqs_cis(dim=16, end=256, rope_base=1e4, rope_scaling=cfg)
        assert torch.allclose(sin[0].float(), torch.zeros(16), atol=1e-5)
