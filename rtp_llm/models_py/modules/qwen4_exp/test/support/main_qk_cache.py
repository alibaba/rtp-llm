from types import SimpleNamespace
from unittest.mock import Mock

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.main_qk_cache_triton import (
    prepare_main_qk_cache,
)
from rtp_llm.models_py.modules.qwen4_exp.test.support.main_qk_norm_rope import (
    MainQkNormRopeFixture,
)
from rtp_llm.ops.fused_rope_kvcache_op import (
    FusedRopeAttnParams,
    _get_fused_rope_kvcache,
)


class MainQkCacheFixture:
    def init_fixture(self):
        self.tester = MainQkNormRopeFixture()

    def case(self, heads, batch, query, prefixes, pool_pages=None):
        cfg, norm, source, positions, params, original, plain, _ = (
            self.tester.setup_case(heads, batch * query)
        )
        columns = (max(prefixes) + query + 127) // 128
        pages = pool_pages or (1 + batch * columns)
        table = torch.arange(
            1, 1 + batch * columns, device="cuda", dtype=torch.int32
        ).view(batch, columns)
        if pool_pages is not None:
            table.fill_(pool_pages - 1)
        cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * query
        length = torch.full((batch,), query, dtype=torch.int32).pin_memory()
        prefix = torch.tensor(prefixes, dtype=torch.int32).pin_memory()
        params = FusedRopeAttnParams(
            _get_fused_rope_kvcache().convert_offset_to_block_array(table),
            None,
            None,
            positions.flatten(),
            cu,
            cu,
            length,
            prefix,
            length,
            query,
            max(prefixes),
            sum(prefixes) + batch * query,
            False,
            torch.bfloat16,
        )
        lens = (
            torch.tensor(prefixes, device="cuda", dtype=torch.int32)[:, None]
            + torch.arange(1, query + 1, device="cuda", dtype=torch.int32)
        ).contiguous()
        if pool_pages is None:
            raw = (torch.randn(pages, 2, 1, 128, 256, device="cuda") * 0.2).to(
                torch.float8_e4m3fn
            )
        else:
            raw = torch.empty(
                pages, 2, 1, 128, 256, device="cuda", dtype=torch.float8_e4m3fn
            )
            raw.view(torch.uint8).zero_()
        scales = torch.rand(pages, 256, device="cuda") + 0.5
        cache = SimpleNamespace(kv_cache_base=raw, kv_scale_base=scales)
        return cfg, norm, source, positions, params, original, table, lens, cache

    def prepared(self, c, cache):
        cfg, norm, source, positions, params, _, table, lens, _ = c
        plan = SimpleNamespace(kv_lens=lens, block_table=table)
        return prepare_main_qk_cache(
            source,
            positions,
            norm,
            cfg.rope_config,
            cache.kv_cache_base,
            cache.kv_scale_base,
            plan,
            params,
        )

    def mutation_owner(self, c):
        cfg, norm, source, positions, params, original, table, lens, cache = c
        return SimpleNamespace(
            _selected_indices=None,
            _paged_kv_scale_args=Mock(return_value={}),
            _is_draft_graph=lambda: False,
            _is_prefix_reuse_prefill=lambda: False,
            is_target_verify=True,
            is_prefill=True,
            _prepare_paged_inputs=Mock(
                return_value=SimpleNamespace(kv_lens=lens, block_table=table)
            ),
            rope_kvcache_impl=Mock(),
            rope_params=params,
            attn_configs=cfg,
            _qsa_main_cache_mutation_started=False,
        )
