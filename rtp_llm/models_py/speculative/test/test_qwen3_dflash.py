import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.model_factory import ModelFactory
from rtp_llm.model_factory_register import ModelDict, ensure_model_registered
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.qwen_3_dflash import Qwen3DFlash, Qwen3DFlashWeight
from rtp_llm.models_py.speculative.dspark_proposer_mixin import DSparkProposerMixin
from rtp_llm.models_py.triton_kernels.common.dflash_attention import is_supported
from rtp_llm.ops import DataType, KvCacheDataType
from rtp_llm.ops.compute_ops import PyAttentionInputs
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity


def _reference_rtp_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """RTP fused Q/K RMSNorm: FP32 math followed by BF16 nearest-even."""
    return (
        x.float()
        * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
        * weight.float()
    ).to(x.dtype)


def _uses_gfx942_aiter_truncate(tensor: torch.Tensor) -> bool:
    if not tensor.is_cuda or not torch.version.hip:
        return False
    arch = getattr(torch.cuda.get_device_properties(tensor.device), "gcnArchName", "")
    return str(arch).startswith("gfx942")


def _aiter_bf16_output(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Model the verified Opus BF16 store used by this AITER gfx942 build."""
    if not _uses_gfx942_aiter_truncate(like):
        return x.to(like.dtype)
    # OPUS_FP32_to_BF16_DEFAULT=2 retains the high FP32 16 bits.  This is
    # truncation toward zero for finite values, unlike torch's RNE conversion.
    high_bits = x.float().contiguous().view(torch.int32) & -65536
    return high_bits.view(torch.float32).to(like.dtype)


def _reference_aiter_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    value = (
        x.float()
        * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
        * weight.float()
    )
    return _aiter_bf16_output(value, x)


def _reference_aiter_silu_and_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return _aiter_bf16_output(torch.nn.functional.silu(gate.float()) * up.float(), gate)


def _reference_dflash_hidden(
    model, weights, features, query_ids, prefix, context_start=0
):
    """Dense torch oracle independent of the model's paged-attention path."""
    attn = model.attn_configs
    q_size = attn.head_num * attn.size_per_head
    kv_size = attn.kv_head_num * attn.size_per_head
    context_rows = features.shape[0]
    width = query_ids.numel()
    hidden = model.embed_tokens(query_ids.reshape(-1))
    context = _reference_aiter_rms_norm(
        features @ weights.global_weights[W.dspark_fc_w],
        weights.global_weights[W.dspark_hidden_norm_gamma],
        model.config.layernorm_eps,
    )
    positions = prefix + torch.arange(width, device=hidden.device, dtype=torch.int32)
    context_positions = torch.arange(
        context_start,
        context_start + context_rows,
        device=hidden.device,
        dtype=torch.int32,
    )
    context_kv = context @ torch.cat(
        [layer_weights[W.attn_qkv_w][:, q_size:] for layer_weights in weights.weights],
        dim=1,
    )
    for index, layer_weights in enumerate(weights.weights):
        pre = _reference_aiter_rms_norm(
            hidden, layer_weights[W.pre_ln_gamma], model.config.layernorm_eps
        )
        qkv = pre @ layer_weights[W.attn_qkv_w]
        query = qkv[:, :q_size].view(width, attn.head_num, attn.size_per_head)
        key = qkv[:, q_size : q_size + kv_size].view(
            width, attn.kv_head_num, attn.size_per_head
        )
        value = qkv[:, q_size + kv_size :].view(
            width, attn.kv_head_num, attn.size_per_head
        )
        # Q/K use RTP fused_qk_rmsnorm, not AITER RMSNorm; it stores RN BF16.
        query = _reference_rtp_rms_norm(
            query, layer_weights[W.q_ln_gamma], model.config.layernorm_eps
        )
        key = _reference_rtp_rms_norm(
            key, layer_weights[W.k_ln_gamma], model.config.layernorm_eps
        )
        context_key = _reference_aiter_rms_norm(
            context_kv[:, index * 2 * kv_size : index * 2 * kv_size + kv_size].view(
                context_rows, attn.kv_head_num, attn.size_per_head
            ),
            layer_weights[W.k_ln_gamma],
            model.config.layernorm_eps,
        )
        context_value = context_kv[
            :, index * 2 * kv_size + kv_size : (index + 1) * 2 * kv_size
        ].view(context_rows, attn.kv_head_num, attn.size_per_head)
        dummy = torch.zeros(
            (context_rows, 1, attn.size_per_head),
            dtype=hidden.dtype,
            device=hidden.device,
        )
        model.context_rope._apply_rope(
            dummy, context_key, SimpleNamespace(positions_d=context_positions)
        )
        model.context_rope._apply_rope(
            query, key, SimpleNamespace(positions_d=positions)
        )
        groups = attn.head_num // attn.kv_head_num
        keys = (
            torch.cat((context_key, key), dim=0)
            .repeat_interleave(groups, dim=1)
            .float()
        )
        values = (
            torch.cat((context_value, value), dim=0)
            .repeat_interleave(groups, dim=1)
            .float()
        )
        scores = torch.einsum("qhd,khd->qhk", query.float(), keys) * (
            attn.size_per_head**-0.5
        )
        key_positions = torch.cat((context_positions, positions), dim=0)
        if model.layers[index].self_attn.layer_type == "sliding_attention":
            visible = key_positions.unsqueeze(0) <= positions.unsqueeze(1)
            visible &= (
                key_positions.unsqueeze(0)
                > positions.unsqueeze(1) - model.layers[index].self_attn.window_size
            )
            scores = scores.masked_fill(~visible.unsqueeze(1), float("-inf"))
        attended = torch.einsum("qhk,khd->qhd", scores.softmax(-1), values).to(
            hidden.dtype
        )
        hidden = hidden + attended.reshape(width, q_size) @ layer_weights[W.attn_o_w]
        post = _reference_aiter_rms_norm(
            hidden, layer_weights[W.post_ln_gamma], model.config.layernorm_eps
        )
        mlp_gate = post @ layer_weights[W.ffn_w1]
        mlp_up = post @ layer_weights[W.ffn_w3]
        activated = _reference_aiter_silu_and_mul(mlp_gate, mlp_up)
        hidden = hidden + activated @ layer_weights[W.ffn_w2]
    return _reference_aiter_rms_norm(
        hidden, weights.global_weights[W.final_ln_gamma], model.config.layernorm_eps
    )


class _DFlashProposalInputHarness(DSparkProposerMixin):
    def __init__(self) -> None:
        self.init_dspark_proposer(
            width=7,
            query_width=8,
            noise_token_id=248070,
            aux_feature_dim=1,
            hidden_dim=1,
        )
        self.seen_query_ids = None

    def forward_query_block(
        self,
        query_ids,
        query_positions,
        prefix_lengths,
        active_requests,
        inputs,
        fmha_impl,
    ):
        del query_positions, prefix_lengths, active_requests, inputs, fmha_impl
        self.seen_query_ids = query_ids.clone()
        return torch.zeros((query_ids.numel(), 1), dtype=torch.bfloat16)

    def combine_hidden_states(self, features):
        return features

    def commit_feature_rows(self, *args, **kwargs):
        raise AssertionError("proposal input test must not commit feature rows")


class Qwen3DFlashConfigTest(unittest.TestCase):
    @staticmethod
    def _raw_config() -> dict:
        return {
            "architectures": ["DFlashDraftModel"],
            "model_type": "qwen3",
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 5,
            "head_dim": 128,
            "vocab_size": 248320,
            "block_size": 16,
            "sliding_window": 2048,
            "layer_types": [
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
            "dflash_config": {
                "mask_token_id": 248070,
                "target_layer_ids": [1, 16, 31, 46, 61],
            },
        }

    def test_config_and_registry_keep_native_dflash_metadata(self) -> None:
        raw = self._raw_config()
        with tempfile.TemporaryDirectory() as path:
            Path(path, "config.json").write_text(json.dumps(raw))
            config = Qwen3DFlash._create_config(path)

        self.assertFalse(config.attn_config.is_causal)
        self.assertEqual(config.input_vocab_size, 248320)
        self.assertEqual(config.dflash_mask_token_id, 248070)
        self.assertEqual(config.dflash_target_layer_ids, [1, 16, 31, 46, 61])
        self.assertEqual(config.dflash_sliding_window, 2048)
        self.assertEqual(config.dflash_native_block_size, 16)
        self.assertEqual(
            config.dflash_layer_types,
            [
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
        )
        self.assertTrue(ensure_model_registered("qwen_3_dflash"))
        self.assertEqual(ModelDict.get_ft_model_type_by_config(raw), "qwen_3_dflash")

    def test_config_accepts_checkpoint_nested_block_metadata(self) -> None:
        raw = self._raw_config()
        raw.pop("block_size")
        raw["dflash_config"]["block_size"] = 16
        raw["rope_parameters"] = {"rope_theta": 10_000_000}
        with tempfile.TemporaryDirectory() as path:
            Path(path, "config.json").write_text(json.dumps(raw))
            config = Qwen3DFlash._create_config(path)

        self.assertEqual(config.dflash_native_block_size, 16)
        self.assertEqual(config.attn_config.rope_config.base, 10_000_000)

    def test_setup_uses_effective_model_config_dtype_and_resets_inherited_fp8_kv(
        self,
    ) -> None:
        raw = self._raw_config()
        raw["dtype"] = "bfloat16"
        with tempfile.TemporaryDirectory() as path:
            Path(path, "config.json").write_text(json.dumps(raw))
            draft_config = Qwen3DFlash._create_config(path)
            draft_config.init_precision_config(
                SimpleNamespace(fp8_kv_cache=True), act_type=None
            )

        self.assertEqual(draft_config.data_type, DataType.TYPE_BF16)
        self.assertEqual(draft_config.attn_config.kv_cache_dtype, KvCacheDataType.FP8)
        target_config = SimpleNamespace(
            num_layers=64,
            hidden_size=5120,
            vocab_size=248320,
            input_vocab_size=248320,
            capture_aux_hidden_layer_ids=None,
        )
        sp_config = SimpleNamespace(gen_num_per_cycle=7)
        ModelFactory._setup_dflash_configs(sp_config, target_config, draft_config)
        self.assertEqual(draft_config.attn_config.kv_cache_dtype, KvCacheDataType.BASE)

    def test_setup_wires_fixed_block_compatibility_without_markov(self) -> None:
        sp_config = SimpleNamespace(
            gen_num_per_cycle=7,
            sp_dspark_mask_token_id=-1,
            sp_dspark_sample_from_anchor=True,
        )
        target_config = SimpleNamespace(
            num_layers=64,
            hidden_size=5120,
            vocab_size=248320,
            capture_aux_hidden_layer_ids=None,
        )
        draft_config = SimpleNamespace(
            num_layers=5,
            hidden_size=5120,
            vocab_size=248320,
            input_vocab_size=248320,
            dflash_mask_token_id=248070,
            dflash_target_layer_ids=[1, 16, 31, 46, 61],
            dflash_layer_types=[
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
            dflash_sliding_window=2048,
            dflash_native_block_size=16,
            config_dtype="bfloat16",
            data_type=DataType.TYPE_BF16,
            quantization=None,
            qk_norm=True,
            attn_config=SimpleNamespace(kv_cache_dtype=KvCacheDataType.FP8),
            capture_aux_hidden_layer_ids=None,
        )

        ModelFactory._setup_dflash_configs(sp_config, target_config, draft_config)

        self.assertEqual(sp_config.sp_dspark_mask_token_id, 248070)
        self.assertFalse(sp_config.sp_dspark_sample_from_anchor)
        self.assertEqual(
            target_config.capture_aux_hidden_layer_ids, [1, 16, 31, 46, 61]
        )
        self.assertEqual(draft_config.capture_aux_hidden_layer_ids, [1, 16, 31, 46, 61])
        self.assertEqual(draft_config.attn_config.kv_cache_dtype, KvCacheDataType.BASE)

    def test_setup_rejects_mask_outside_full_input_vocab(self) -> None:
        sp_config = SimpleNamespace(gen_num_per_cycle=7)
        target_config = SimpleNamespace(
            num_layers=64,
            hidden_size=5120,
            vocab_size=248320,
            capture_aux_hidden_layer_ids=None,
        )
        draft_config = SimpleNamespace(
            num_layers=5,
            hidden_size=5120,
            vocab_size=248320,
            input_vocab_size=248320,
            dflash_mask_token_id=248320,
            dflash_target_layer_ids=[1, 16, 31, 46, 61],
            dflash_layer_types=["sliding_attention"] * 4 + ["full_attention"],
            dflash_sliding_window=2048,
            dflash_native_block_size=16,
            config_dtype="bfloat16",
            data_type=DataType.TYPE_BF16,
            quantization=None,
            qk_norm=True,
            attn_config=SimpleNamespace(kv_cache_dtype=KvCacheDataType.BASE),
            capture_aux_hidden_layer_ids=None,
        )

        with self.assertRaisesRegex(ValueError, "dflash_mask_token_id"):
            ModelFactory._setup_dflash_configs(sp_config, target_config, draft_config)

    def test_setup_rejects_unsupported_query_width(self) -> None:
        sp_config = SimpleNamespace(gen_num_per_cycle=8)
        target_config = SimpleNamespace(
            num_layers=64,
            hidden_size=5120,
            vocab_size=248320,
            capture_aux_hidden_layer_ids=None,
        )
        draft_config = SimpleNamespace(
            num_layers=5,
            hidden_size=5120,
            vocab_size=248320,
            input_vocab_size=248320,
            dflash_mask_token_id=248070,
            dflash_target_layer_ids=[1, 16, 31, 46, 61],
            dflash_layer_types=["sliding_attention"] * 4 + ["full_attention"],
            dflash_sliding_window=2048,
            dflash_native_block_size=16,
            config_dtype="bfloat16",
            data_type=DataType.TYPE_BF16,
            quantization=None,
            qk_norm=True,
            attn_config=SimpleNamespace(kv_cache_dtype=KvCacheDataType.BASE),
            capture_aux_hidden_layer_ids=None,
        )

        with self.assertRaisesRegex(ValueError, "gamma=1..7"):
            ModelFactory._setup_dflash_configs(sp_config, target_config, draft_config)

    def test_loader_declares_borrowed_vocab_tensors_and_dflash_globals(self) -> None:
        base_info = ModelWeightInfo(
            [
                AtomicWeight(
                    W.embedding,
                    [CkptWeightInfo("embed_tokens.weight", identity)],
                    identity,
                ),
                AtomicWeight(
                    W.lm_head, [CkptWeightInfo("lm_head.weight", identity)], identity
                ),
                AtomicWeight(
                    W.final_ln_gamma,
                    [CkptWeightInfo("norm.weight", identity)],
                    identity,
                ),
            ],
            [],
        )
        with patch(
            "rtp_llm.models.qwen_3_dflash.QWenV3Weight._get_weight_info",
            return_value=base_info,
        ):
            # ``super()`` needs an instance compatible with the concrete
            # loader class even though the mocked parent method does not read
            # instance state.
            info = Qwen3DFlashWeight._get_weight_info(
                Qwen3DFlashWeight.__new__(Qwen3DFlashWeight)
            )

        names = [weight.name for weight in info.weights]
        self.assertIn(W.embedding, names)
        self.assertIn(W.lm_head, names)
        self.assertIn(W.final_ln_gamma, names)
        self.assertIn(W.dspark_fc_w, names)
        self.assertIn(W.dspark_hidden_norm_gamma, names)

    def test_production_alias_contract_requires_identical_full_vocab(self) -> None:
        target = SimpleNamespace(
            model_config=SimpleNamespace(
                hidden_size=5120,
                vocab_size=248320,
                data_type=DataType.TYPE_BF16,
                enable_fp32_lm_head=True,
            )
        )
        draft = SimpleNamespace(
            hidden_size=5120,
            vocab_size=248320,
            data_type=DataType.TYPE_BF16,
            enable_fp32_lm_head=True,
        )
        self.assertEqual(
            Qwen3DFlash.speculative_weight_alias_names(target, draft),
            (W.embedding, W.lm_head),
        )
        draft.vocab_size = 20000
        with self.assertRaisesRegex(ValueError, "incompatible"):
            Qwen3DFlash.speculative_weight_alias_names(target, draft)

    def test_propose_input_keeps_mask_valued_anchor_and_masks_only_tail(self) -> None:
        harness = _DFlashProposalInputHarness()
        attention = PyAttentionInputs()
        attention.prefix_lengths = torch.tensor([17, 23], dtype=torch.int32)
        inputs = SimpleNamespace(
            attention_inputs=attention,
            input_ids=torch.tensor(
                [
                    248070,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    42,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                ],
                dtype=torch.int32,
            ),
        )

        harness.run_propose_step(inputs, fmha_impl=None, device=torch.device("cpu"))

        self.assertIsNotNone(harness.seen_query_ids)
        self.assertEqual(harness.seen_query_ids[:, 0].tolist(), [248070, 42])
        self.assertTrue(
            torch.equal(harness.seen_query_ids[:, 1:], torch.full((2, 7), 248070))
        )

    def test_dflash_kernel_gate_rejects_cpu_and_unsupported_dtype(self) -> None:
        query = torch.empty((8, 2, 128), dtype=torch.bfloat16)
        cache = torch.empty((2, 2, 1, 16, 128), dtype=torch.bfloat16)
        self.assertFalse(is_supported(query, cache, 8))
        self.assertFalse(
            is_supported(query.to(torch.float16), cache.to(torch.float16), 8)
        )

    def test_prepare_skips_large_commit_attention_output_reserve(self) -> None:
        from rtp_llm.models_py.model_desc.qwen3_dflash_model import Qwen3DFlashModel

        model = Qwen3DFlashModel.__new__(Qwen3DFlashModel)
        torch.nn.Module.__init__(model)
        reserved_rows = []
        attention = SimpleNamespace(
            _out=torch.empty((8, 2, 64), dtype=torch.bfloat16),
            reserve_output=lambda rows, _device, _dtype: reserved_rows.append(rows),
        )
        model.layers = [SimpleNamespace(self_attn=attention)]
        model.embed_tokens = SimpleNamespace(
            weight=torch.empty((1, 1), dtype=torch.bfloat16)
        )

        commit_inputs = SimpleNamespace(
            input_ids=torch.empty(65_536, dtype=torch.int32),
            input_hiddens=torch.empty((65_536, 768), dtype=torch.bfloat16),
        )
        model.prepare_fmha_impl(commit_inputs)
        self.assertEqual(reserved_rows, [])
        self.assertEqual(tuple(attention._out.shape), (8, 2, 64))

        propose_inputs = SimpleNamespace(
            input_ids=torch.empty(8, dtype=torch.int32),
            input_hiddens=torch.empty(0, dtype=torch.bfloat16),
        )
        model.prepare_fmha_impl(propose_inputs, is_cuda_graph=True)
        self.assertEqual(reserved_rows, [8])

        # The graph runner allocates a stable nonempty hidden buffer for every
        # graph role, including proposal.  It must not suppress the proposal
        # output allocation merely because that buffer exists.
        graph_propose_inputs = SimpleNamespace(
            input_ids=torch.empty(8, dtype=torch.int32),
            input_hiddens=torch.empty((8, 768), dtype=torch.bfloat16),
        )
        model.prepare_fmha_impl(graph_propose_inputs, is_cuda_graph=True)
        self.assertEqual(reserved_rows, [8, 8])

    def test_gpu_randomweight_model_commit_and_mixed_block_q8_q16(self) -> None:
        """Exercise the production model, not a toy attention implementation."""
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA or ROCm for production Triton attention")

        from rtp_llm.model_loader.model_weight_info import ModelWeights
        from rtp_llm.models_py.model_desc.qwen3_dflash_model import Qwen3DFlashModel
        from rtp_llm.ops import ParallelismConfig

        device = torch.device("cuda")
        torch.manual_seed(17)

        class TinyKVCache:
            def __init__(self, layer_caches):
                self.layer_caches = layer_caches

            def get_layer_cache(self, index):
                return self.layer_caches[index]

            def get_layer_cache_groups(self, index):
                return [self.layer_caches[index]]

        for head_num, kv_head_num in ((2, 1), (4, 2)):
            for width in (8, 16):
                raw = {
                    "architectures": ["DFlashDraftModel"],
                    "hidden_size": 192,
                    "intermediate_size": 256,
                    "num_attention_heads": head_num,
                    "num_key_value_heads": kv_head_num,
                    "num_hidden_layers": 3,
                    "head_dim": 64,
                    "vocab_size": 64,
                    "dtype": "bfloat16",
                    "rope_theta": 10_000,
                    "block_size": 16,
                    "sliding_window": 4,
                    "layer_types": [
                        "sliding_attention",
                        "sliding_attention",
                        "full_attention",
                    ],
                    "dflash_config": {
                        "mask_token_id": 31,
                        "target_layer_ids": [0, 1, 2, 3],
                    },
                }
                with tempfile.TemporaryDirectory() as path:
                    Path(path, "config.json").write_text(json.dumps(raw))
                    config = Qwen3DFlash._create_config(path)
                config.gen_num_per_cycle = width - 1
                config.max_seq_len = 64
                par = ParallelismConfig()
                weights = ModelWeights(3, str(device), torch.bfloat16)

                def random(shape):
                    return (
                        torch.randn(*shape, device=device, dtype=torch.bfloat16) * 0.02
                    )

                def norm(shape):
                    return torch.ones(
                        *shape, device=device, dtype=torch.bfloat16
                    ) + random(shape)

                weights.set_global_weight(W.embedding, random((64, 192)))
                target_head = random((64, 192))
                weights.set_global_weight(W.lm_head, target_head)
                weights.set_global_weight(W.final_ln_gamma, norm((192,)))
                weights.set_global_weight(W.dspark_fc_w, random((768, 192)))
                weights.set_global_weight(W.dspark_hidden_norm_gamma, norm((192,)))
                for layer_index in range(3):
                    layer = weights.weights[layer_index]
                    layer[W.pre_ln_gamma] = norm((192,))
                    layer[W.post_ln_gamma] = norm((192,))
                    layer[W.q_ln_gamma] = norm((64,))
                    layer[W.k_ln_gamma] = norm((64,))
                    layer[W.attn_qkv_w] = random(
                        (192, (head_num + 2 * kv_head_num) * 64)
                    )
                    layer[W.attn_o_w] = random((head_num * 64, 192))
                    layer[W.ffn_w1] = random((192, 256))
                    layer[W.ffn_w3] = random((192, 256))
                    layer[W.ffn_w2] = random((256, 192))

                model = Qwen3DFlashModel(
                    config, par, weights, max_generate_batch_size=1
                )
                caches = [
                    SimpleNamespace(
                        kv_cache_base=torch.zeros(
                            8,
                            2,
                            kv_head_num,
                            16,
                            64,
                            device=device,
                            dtype=torch.bfloat16,
                        )
                    )
                    for _ in range(3)
                ]
                model.kv_cache = TinyKVCache(caches)
                table = torch.arange(8, device=device, dtype=torch.int32).view(1, 8)
                attention = PyAttentionInputs()
                attention.kv_cache_kernel_block_id_device = table
                attention.kv_cache_kernel_block_id = table
                inputs = SimpleNamespace(
                    attention_inputs=attention,
                    input_ids=torch.full(
                        (width,), 31, device=device, dtype=torch.int32
                    ),
                )
                model.prepare_fmha_impl(inputs)

                features = random((5, 768))
                main_x = model.combine_hidden_states(features)
                captured_context = []
                hook = model.context_kv_projection.register_forward_pre_hook(
                    lambda _module, args: captured_context.append(
                        args[0].detach().clone()
                    )
                )
                try:
                    model.commit_feature_rows(
                        main_x,
                        torch.zeros(5, device=device, dtype=torch.int32),
                        torch.arange(5, device=device, dtype=torch.int32),
                        torch.tensor([5], device=device, dtype=torch.int32),
                        inputs,
                    )
                finally:
                    hook.remove()
                self.assertEqual(len(captured_context), 1)
                torch.testing.assert_close(
                    captured_context[0], model.hidden_norm(main_x), atol=2e-2, rtol=2e-2
                )
                committed_nonzero = [
                    int(cache.kv_cache_base.count_nonzero()) for cache in caches
                ]

                query_ids = torch.full((1, width), 31, device=device, dtype=torch.int32)
                hidden = model.forward_query_block(
                    query_ids,
                    5 + torch.arange(width, device=device).view(1, width),
                    torch.tensor([5], device=device, dtype=torch.int32),
                    torch.tensor([True], device=device),
                    inputs,
                    fmha_impl=None,
                )
                expected_hidden = _reference_dflash_hidden(
                    model,
                    weights,
                    features,
                    query_ids,
                    prefix=5,
                )
                torch.testing.assert_close(
                    hidden, expected_hidden, atol=3e-2, rtol=3e-2
                )
                logits = hidden.float() @ target_head.float().T
                expected_logits = expected_hidden.float() @ target_head.float().T
                torch.testing.assert_close(
                    logits, expected_logits, atol=3e-2, rtol=3e-2
                )
                self.assertEqual(tuple(logits.shape), (width, 64))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertTrue(
                    all(
                        int(cache.kv_cache_base.count_nonzero()) > before
                        for cache, before in zip(caches, committed_nonzero)
                    )
                )

                if width != 8:
                    continue

                # Capture the real role entrypoints with a continuous context: an eager seed, then a graph-owned
                # 5-row commit. Replay grows the seed from 8 to 16 rows.
                # PyAttentionInputs exposes C++-owned *_device mirrors as
                # read-only.  The direct Python model test supplies their
                # graph-stable CUDA source tensors through the writable fields;
                # device_metadata_tensor then uses those tensors as its fallback.
                for cache in caches:
                    cache.kv_cache_base.zero_()
                prefix_lengths = torch.zeros(1, device=device, dtype=torch.int32)
                input_lengths = torch.zeros(1, device=device, dtype=torch.int32)
                next_prefix = torch.full((1,), 13, device=device, dtype=torch.int32)
                attention.prefix_lengths = prefix_lengths
                attention.input_lengths = input_lengths
                seed_features = random((16, 768))
                graph_features = random((5, 768))
                seed_inputs = SimpleNamespace(
                    input_ids=torch.zeros(16, device=device, dtype=torch.int32),
                    input_hiddens=seed_features,
                    attention_inputs=attention,
                )
                commit_inputs = SimpleNamespace(
                    input_ids=torch.zeros(5, device=device, dtype=torch.int32),
                    input_hiddens=graph_features,
                    attention_inputs=attention,
                )
                graph_anchor_ids = torch.full(
                    (8,), 31, device=device, dtype=torch.int32
                )
                graph_anchor_ids[0] = 7
                propose_inputs = SimpleNamespace(
                    input_ids=graph_anchor_ids,
                    attention_inputs=attention,
                )

                model.prepare_fmha_impl(propose_inputs, is_cuda_graph=True)
                # Warm up all compilation/allocation paths outside capture.
                prefix_lengths.zero_()
                input_lengths.fill_(8)
                model.forward_commit(seed_inputs)
                prefix_lengths.fill_(8)
                input_lengths.fill_(5)
                model.forward_commit(commit_inputs)
                prefix_lengths.copy_(next_prefix)
                model.forward_propose(propose_inputs)
                torch.cuda.synchronize()

                # The capture contains only the tail commit and proposal.  The
                # preceding eight KV rows persist and are valid in both masks.
                prefix_lengths.fill_(8)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph), torch.inference_mode():
                    model.forward_commit(commit_inputs)
                    prefix_lengths.copy_(next_prefix)
                    graph_propose_output = model.forward_propose(propose_inputs)

                replay_features = random((5, 768))
                replay_anchor_ids = torch.full(
                    (8,), 31, device=device, dtype=torch.int32
                )
                replay_anchor_ids[0] = 19
                replay_table = torch.tensor(
                    [[3, 2, 1, 0, 4, 5, 6, 7]], device=device, dtype=torch.int32
                )
                commit_inputs.input_hiddens.copy_(replay_features)
                propose_inputs.input_ids.copy_(replay_anchor_ids)
                table.copy_(replay_table)
                next_prefix.fill_(21)
                # A remapped table needs its persistent seed rewritten under the
                # new page IDs before the captured tail operation is replayed.
                for cache in caches:
                    cache.kv_cache_base.zero_()
                prefix_lengths.zero_()
                input_lengths.fill_(16)
                model.forward_commit(seed_inputs)
                prefix_lengths.fill_(16)
                input_lengths.fill_(5)
                graph.replay()
                graph_hidden = graph_propose_output.hidden_states.clone()

                # Fresh eager cache with exactly the replay's seed/table/metadata;
                # this prevents preceding warmup or capture writes from leaking into
                # the comparison.
                for cache in caches:
                    cache.kv_cache_base.zero_()
                prefix_lengths.zero_()
                input_lengths.fill_(16)
                model.forward_commit(seed_inputs)
                prefix_lengths.fill_(16)
                input_lengths.fill_(5)
                model.forward_commit(commit_inputs)
                prefix_lengths.copy_(next_prefix)
                eager_hidden = model.forward_propose(
                    propose_inputs
                ).hidden_states.clone()
                reference_query_ids = torch.full(
                    (1, 8), 31, device=device, dtype=torch.int32
                )
                reference_query_ids[0, 0] = 19
                reference_hidden = _reference_dflash_hidden(
                    model,
                    weights,
                    torch.cat((seed_features, replay_features), dim=0),
                    reference_query_ids,
                    prefix=21,
                )
                torch.testing.assert_close(
                    graph_hidden, eager_hidden, atol=3e-2, rtol=3e-2
                )
                torch.testing.assert_close(
                    graph_hidden, reference_hidden, atol=3e-2, rtol=3e-2
                )


if __name__ == "__main__":
    unittest.main()
