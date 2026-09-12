"""模型 config 解析回归测试（守住 HF config 新旧格式的兼容）。

背景：transformers>=5.2.0 调整了 config.json 的 schema
- rope_theta 从顶层移入嵌套的 rope_parameters
- torch_dtype 改名为 dtype
若解析器只按老格式读顶层字段，会静默回退默认值（rope base -> 10000, dtype -> fp16），
导致长上下文行为系统性劣化。此文件用于覆盖新旧两种格式的解析。
"""

import unittest

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.qwen_v2 import QWenV2

# 最小必填字段（满足 QWenV2._from_config_json 的读取），值取自 Qwen3-8B。
_BASE_CONFIG = {
    "intermediate_size": 12288,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "hidden_size": 4096,
    "num_hidden_layers": 36,
    "vocab_size": 151936,
    "rms_norm_eps": 1e-06,
}


def _parse(extra):
    cfg = ModelConfig()
    config_json = dict(_BASE_CONFIG)
    config_json.update(extra)
    QWenV2._from_config_json(cfg, config_json)
    return cfg


class QwenConfigParseTest(unittest.TestCase):
    def test_transformers_5_2_nested_rope_and_dtype(self):
        # transformers>=5.2.0 新格式：rope_theta 在 rope_parameters 内，精度字段叫 dtype
        cfg = _parse(
            {
                "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
                "dtype": "bfloat16",
            }
        )
        # 修复前这里会是 10000（找不到顶层 rope_theta 而静默回退）
        self.assertEqual(cfg.attn_config.rope_config.base, 1000000)
        self.assertNotEqual(cfg.attn_config.rope_config.base, 10000)
        self.assertEqual(cfg.config_dtype, "bfloat16")

    def test_legacy_flat_rope_and_torch_dtype(self):
        # 老格式（<=4.x）：顶层 rope_theta / torch_dtype，修复后仍需正常
        cfg = _parse(
            {
                "rope_theta": 1000000,
                "torch_dtype": "bfloat16",
            }
        )
        self.assertEqual(cfg.attn_config.rope_config.base, 1000000)
        self.assertEqual(cfg.config_dtype, "bfloat16")

    def test_missing_rope_theta_keeps_default(self):
        # 既无顶层 rope_theta 也无 rope_parameters：应保持 ModelConfig 默认值，不报错
        default_base = ModelConfig().attn_config.rope_config.base
        cfg = _parse({})
        self.assertEqual(cfg.attn_config.rope_config.base, default_base)
        self.assertEqual(default_base, 10000)  # 记录当前默认值

    def test_toplevel_rope_theta_takes_precedence(self):
        # 顶层与嵌套同时存在时，顶层优先（老格式优先，避免新旧混写时被覆盖）
        cfg = _parse(
            {
                "rope_theta": 500000,
                "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
            }
        )
        self.assertEqual(cfg.attn_config.rope_config.base, 500000)


if __name__ == "__main__":
    unittest.main()
