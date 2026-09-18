"""Public-contract lint for the dsv41 module surface.

Seed set (stage-R first batch): the import surface consumed by external callers
(``model_desc/deepseek_v41_model.py``, ``model_desc/deepseek_v41_dspark_model.py``,
``models/deepseek_v41.py``, ``model_loader/host_shared_cuda.py``,
``models/multimodal/deepseek_v41_vision.py``) plus the C++ pybind consumption
point (``rtp_llm_ops.rmsnorm``). ``modules/dsv41/__init__.py`` re-exports
nothing, so the contract is the per-module attribute surface below.

Refactor units must keep this surface intact, or update this seed in the same
commit that changes the corresponding consumers.
"""

import unittest

from standalone_load import load_component

# module file (under models_py/modules/dsv41/) -> public symbols used externally
CONTRACT = {
    "_engram_lookup_triton.py": [
        "engram_gather_kernel",
        "engram_gather_quantize_kernel",
    ],
    "_vision_fa4.py": ["vision_attention_fa4"],
    "_vision_rope_triton.py": ["apply_vision_qk_rope"],
    "attention.py": ["AttentionOwnerCache", "V41Attention", "V41AttentionCache"],
    "cache_layout.py": [
        "CacheLayout",
        "CacheRegion",
        "GLOBAL_OWNERS",
        "PAIR_OWNERS",
        "RegionSlot",
        "layer_sources",
    ],
    "ced.py": ["ReplayConfig", "ReplayMode"],
    "compact_reader.py": ["CompactPages", "GlobalBinding", "SwaBinding"],
    "compressor.py": ["PairCarry"],
    "cp.py": ["begin_cp_request"],
    "decode_attention.py": ["_same_pages"],
    "decode_compressor.py": ["_tensor", "normalize_empty_pair_checkpoint"],
    "decode_draft.py": ["V41DraftAttentionBuffers", "V41DraftModel"],
    "decode_fmha_impl.py": ["V41DecodeFmhaImpl", "_copy_bytes", "_host"],
    "draft.py": ["V41PrefillDraftCommit"],
    "indexer.py": ["warmup_sparse_indexer"],
    "inputs.py": ["V41ModelRows"],
    "linear.py": ["warmup_block32_linears"],
    "math.py": ["dequantize_block32"],
    "moe.py": ["V41MoE"],
    "prefill.py": ["V41CPHistory", "V41CPPrefillExecutor"],
    "transformer.py": ["V41ImageFeatures", "V41TargetModel"],
}

# C++ pybind symbols the dsv41 modules consume (lazy imports inside functions).
PYBIND_CONTRACT = {"rtp_llm.ops.compute_ops": {"rtp_llm_ops": ["rmsnorm"]}}


class ContractLintTest(unittest.TestCase):
    def test_public_symbol_surface(self):
        for relative, symbols in sorted(CONTRACT.items()):
            name = "rtp_v41_contract_" + relative.replace(".py", "").replace("/", "_")
            module = load_component(name, "models_py/modules/dsv41/" + relative)
            for symbol in symbols:
                self.assertTrue(
                    hasattr(module, symbol),
                    f"{relative} lost public symbol {symbol}",
                )

    def test_init_reexports_nothing(self):
        module = load_component(
            "rtp_v41_contract_init", "models_py/modules/dsv41/__init__.py"
        )
        leaked = [name for name in vars(module) if not name.startswith("_")]
        self.assertEqual(leaked, [])

    def test_pybind_consumption_points(self):
        try:
            from rtp_llm.ops.compute_ops import rtp_llm_ops
        except ImportError as exc:
            self.skipTest(f"compute ops extension unavailable: {exc}")
        for attr in PYBIND_CONTRACT["rtp_llm.ops.compute_ops"]["rtp_llm_ops"]:
            self.assertTrue(
                hasattr(rtp_llm_ops, attr), f"rtp_llm_ops lost symbol {attr}"
            )


if __name__ == "__main__":
    unittest.main()
