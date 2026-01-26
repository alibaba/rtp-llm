import random
from typing import Dict, Optional
from unittest import SkipTest, TestCase, main

import torch

device = torch.device("cuda")

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.modules.hybrid.indexer import Indexer
from rtp_llm.ops.compute_ops import PyAttentionInputs
from rtp_llm.utils.model_weight import W

from .indexer_ref import IndexerRef


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def assert_equal(
    score: torch.Tensor,
    indices_ref: torch.Tensor,
    indices_our: torch.Tensor,
    bs: int,
    k: int,
    seq_len: int,
    topk_indices_offset: Optional[torch.Tensor] = None,
    max_permit_error: int = 0,
):
    indices_our_cpu = indices_our.cpu().tolist()
    indices_ref_cpu = indices_ref.cpu().tolist()

    wrong_values = 0
    for i in range(bs):
        indices_ref_set_i = set(indices_ref_cpu[i])
        indices_our_set_i = set(indices_our_cpu[i])
        more = indices_our_set_i - indices_ref_set_i
        less = indices_ref_set_i - indices_our_set_i
        offset = topk_indices_offset[i].item() if topk_indices_offset is not None else 0
        if len(more) > 0 or len(less) > 0:
            # check whether more values are the same with less values
            # if so, either one is acceptable, since their values are the same
            more_values = sorted(score[i, idx - offset].item() for idx in more)
            less_values = sorted(score[i, idx - offset].item() for idx in less)
            if more_values != less_values:
                wrong_values += len(more)
                print(
                    f"{bs=}, {k=}, {seq_len=}, {i=}, {more=}, {less=} failed, with {more_values=}, {less_values=}"
                )
        assert wrong_values <= max_permit_error, f"{wrong_values=}, {max_permit_error=}"


class MockKVCache:
    """Mock KVCache for testing."""

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        kv_lora_rank: int,
        rope_head_dim: int,
        tokens_per_block: int,
        device: str = "cuda",
    ):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.kv_lora_rank = kv_lora_rank
        self.rope_head_dim = rope_head_dim
        self.tokens_per_block = tokens_per_block
        self.device = device

        # Calculate indexer size (matches implementation)
        index_head_dim = 128
        # TODO: 确认为啥？
        indexer_size = index_head_dim + 4

        # Create mock kv_cache_base
        num_blocks = (max_seq_len + tokens_per_block - 1) // tokens_per_block
        total_tokens = num_blocks * batch_size
        cache_dim = indexer_size

        self.kv_scale_base = torch.randn(
            total_tokens,
            tokens_per_block,
            cache_dim,
            dtype=torch.bfloat16,
            device=device,
        ).to(torch.float8_e4m3fn)


class IndexerTest(TestCase):
    """Test suite for Indexer module."""

    BATCH_SIZES = [1, 2]
    SEQ_LENS = [64, 128]
    PAGE_SIZES = [64]

    def setUp(self) -> None:
        """Set up test environment."""
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device(device)
        set_seed(42)

    def _create_test_config(self, hidden_size: int = 7168) -> ModelConfig:
        """Create a test configuration for Indexer."""
        config = ModelConfig()
        config.attn_config.head_num = 128
        config.max_seq_len = 2048
        config.hidden_size = hidden_size

        # MLA-specific config
        config.attn_config.q_lora_rank = 1536
        config.attn_config.rope_head_dim = 64
        config.attn_config.tokens_per_block = 64
        config.attn_config.use_mla = True

        # Indexer-specific config
        config.attn_config.indexer_head_num = 64
        config.attn_config.indexer_head_dim = 128
        config.attn_config.indexer_topk = 2048

        config.quant_config = None
        config.layernorm_eps = 1e-6

        return config

    def _create_test_weights(self, config: ModelConfig) -> Dict[str, torch.Tensor]:
        """Create test weights for Indexer."""
        weights = {}

        # Indexer wq_b weights
        weights[W.mla_indexer_qb_w] = torch.randn(
            config.attn_config.q_lora_rank,
            config.attn_config.indexer_head_num * config.attn_config.indexer_head_dim,
            dtype=torch.bfloat16,
            device=device,
        )

        # Indexer wk weights
        weights[W.mla_indexer_k_w] = torch.randn(
            config.hidden_size,
            config.attn_config.indexer_head_dim,
            dtype=torch.bfloat16,
            device=device,
        )

        # Indexer k_norm weights
        weights[W.mla_indexer_k_norm_w] = torch.randn(
            config.attn_config.indexer_head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        weights[W.mla_indexer_k_norm_b] = torch.randn(
            config.attn_config.indexer_head_dim,
            dtype=torch.bfloat16,
            device=device,
        )

        # Indexer weights_proj weights
        weights[W.mla_indexer_weights_proj_w] = torch.randn(
            config.hidden_size,
            config.attn_config.indexer_head_num,
            dtype=torch.float32,
            device=device,
        )

        # Rotary embedding cos/sin cache
        max_seq_len = config.max_seq_len
        rope_dim = config.attn_config.rope_head_dim
        cos_sin_cache = torch.randn(
            max_seq_len,
            rope_dim,
            dtype=torch.float32,
            device=device,
        )
        weights[W.rope_cos_sin_cache] = cos_sin_cache

        return weights

    def _create_attention_inputs(
        self, batch_size: int, seq_len: int, is_prefill: bool = True
    ) -> PyAttentionInputs:
        """Create mock attention inputs."""
        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = is_prefill

        if is_prefill:
            # Prefill mode
            input_lengths = torch.tensor(
                [seq_len] * batch_size, dtype=torch.int32, device=torch.device("cpu")
            )
            attn_inputs.input_lengths = input_lengths
            attn_inputs.sequence_lengths = torch.tensor(
                [], dtype=torch.int32, device=torch.device("cpu")
            )

            # cu_seqlens: [0, seq_len, 2*seq_len, ...]
            cu_seqlens = torch.arange(
                0,
                (batch_size + 1) * seq_len,
                seq_len,
                dtype=torch.int32,
                device=device,
            )
            attn_inputs.cu_seqlens = cu_seqlens

            # Create kv_cache_block_id
            page_size = 64
            num_pages = (seq_len + page_size - 1) // page_size
            kv_cache_block_id = torch.zeros(
                [batch_size, num_pages],
                dtype=torch.int32,
                device=torch.device("cpu"),
            )
            for i in range(batch_size):
                kv_cache_block_id[i, :] = torch.arange(
                    i * num_pages,
                    (i + 1) * num_pages,
                    dtype=torch.int32,
                )
            attn_inputs.kv_cache_block_id_host = kv_cache_block_id
            attn_inputs.kv_cache_block_id_device = kv_cache_block_id.to(device)
        else:
            # Decode mode
            sequence_lengths = torch.tensor(
                [seq_len] * batch_size, dtype=torch.int32, device=torch.device("cpu")
            )
            attn_inputs.sequence_lengths = sequence_lengths
            attn_inputs.input_lengths = torch.tensor(
                [], dtype=torch.int32, device=torch.device("cpu")
            )

            # In decode mode, cu_seqlens is just [0, 1, 2, ..., batch_size]
            cu_seqlens = torch.arange(
                0,
                batch_size + 1,
                dtype=torch.int32,
                device=device,
            )
            attn_inputs.cu_seqlens = cu_seqlens

            # Create kv_cache_block_id for decode
            page_size = 64
            num_pages = (seq_len + page_size - 1) // page_size
            kv_cache_block_id = torch.zeros(
                [batch_size, num_pages],
                dtype=torch.int32,
                device=torch.device("cpu"),
            )
            for i in range(batch_size):
                kv_cache_block_id[i, :] = torch.arange(
                    i * num_pages,
                    (i + 1) * num_pages,
                    dtype=torch.int32,
                )
            attn_inputs.kv_cache_block_id_host = kv_cache_block_id
            attn_inputs.kv_cache_block_id_device = kv_cache_block_id.to(device)

        attn_inputs.prefix_lengths = torch.zeros(
            batch_size, dtype=torch.int32, device=torch.device("cpu")
        )

        return attn_inputs

    def _run_indexer_forward_test(
        self, batch_size: int, seq_len: int, is_prefill: bool
    ):
        """Helper method to run full forward test."""
        config = self._create_test_config()
        weights = self._create_test_weights(config)

        indexer = Indexer(
            attn_config=config.attn_config,
            weights=weights,
            global_weights=weights,
            layer_idx=0,
            layernorm_eps=config.layernorm_eps,
            quant_config=config.quant_config,
        )
        indexer_ref = IndexerRef(
            attn_config=config.attn_config,
            weights=weights,
            layer_idx=0,
            layernorm_eps=config.layernorm_eps,
            quant_config=config.quant_config,
        )

        # Create inputs
        attn_inputs = self._create_attention_inputs(batch_size, seq_len, is_prefill)
        indexer.prepare(attn_inputs)
        indexer_ref.prepare(attn_inputs)

        # Create KV cache
        kv_cache = MockKVCache(
            batch_size=batch_size,
            max_seq_len=seq_len,
            kv_lora_rank=config.attn_config.kv_lora_rank,
            rope_head_dim=config.attn_config.rope_head_dim,
            tokens_per_block=config.attn_config.tokens_per_block,
            device=device,
        )

        # Create input tensors
        if not is_prefill:
            seq_len = 1

        hidden_states = torch.randn(
            batch_size * seq_len,
            config.hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        q_lora = torch.randn(
            batch_size * seq_len,
            config.attn_config.q_lora_rank,
            dtype=torch.bfloat16,
            device=device,
        )
        result_ref, score, topk_indices_offset = indexer_ref(
            x=hidden_states,
            qr=q_lora,
            kv_cache=kv_cache,
        )
        result = indexer(
            hidden_states=hidden_states.view(-1, config.hidden_size),
            q_lora=q_lora.view(-1, config.attn_config.q_lora_rank),
            kv_cache=kv_cache,
        )

        dst_page_table_our = torch.sort(result, dim=-1).values
        dst_page_table_ref = torch.sort(result_ref, dim=-1).values

        print(f"is_prefill: {is_prefill}")
        print(f"dst_page_table_ref: {dst_page_table_ref.shape}")
        print(f"dst_page_table_ref: {dst_page_table_ref}")
        print(f"dst_page_table_our: {dst_page_table_our.shape}")
        print(f"dst_page_table_our: {dst_page_table_our}")

        if is_prefill:
            assert_equal(
                score,
                dst_page_table_ref,
                dst_page_table_our,
                batch_size * seq_len,
                config.attn_config.indexer_topk,
                seq_len,
                topk_indices_offset,
                max_permit_error=5,
            )
        else:
            assert_equal(
                score,
                dst_page_table_ref,
                dst_page_table_our,
                batch_size,
                config.attn_config.indexer_topk,
                seq_len,
                max_permit_error=5,
            )

    def test_forward_prefill_mode(self):
        self._run_indexer_forward_test(batch_size=2, seq_len=4096, is_prefill=True)

    def test_forward_decode_mode(self):
        self._run_indexer_forward_test(batch_size=2, seq_len=4096, is_prefill=False)


if __name__ == "__main__":
    main()
