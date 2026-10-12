"""V4.1 cache geometry and global-source bindings, independent of CUDA imports."""

from dataclasses import dataclass
from typing import Any, Mapping

from rtp_llm.config.dsv41_config import dsv41_pp_partition

DSV41_TOKENS_PER_BLOCK = 256


@dataclass(frozen=True)
class V41PageLayout:
    entries: int
    payload_bytes_per_entry: int
    scale_bytes_per_entry: int = 0
    alignment: int = 1

    @property
    def scale_plane_offset(self) -> int:
        return self.entries * self.payload_bytes_per_entry

    @property
    def used_bytes(self) -> int:
        return self.scale_plane_offset + self.entries * self.scale_bytes_per_entry

    @property
    def stride_bytes(self) -> int:
        return (self.used_bytes + self.alignment - 1) // self.alignment * self.alignment

    def entry_offsets(self, entry: int) -> tuple[int, int]:
        """Byte offsets within one page; payload and scales are separate planes."""
        if not 0 <= entry < self.entries:
            raise IndexError(f"entry {entry} is outside page capacity {self.entries}")
        return (
            entry * self.payload_bytes_per_entry,
            self.scale_plane_offset + entry * self.scale_bytes_per_entry,
        )


def dsv41_page_layouts(
    gen_num_per_cycle: int,
    physical_tokens_per_block: int = DSV41_TOKENS_PER_BLOCK,
    kernel_tokens_per_block: int = DSV41_TOKENS_PER_BLOCK,
    kv_cache_sharded: bool = False,
) -> dict[str, V41PageLayout]:
    if physical_tokens_per_block != 256 or kernel_tokens_per_block != 256:
        raise ValueError("V4.1 exact cache requires physical/kernel pages 256/256")
    if kv_cache_sharded:
        raise ValueError("V4.1 exact cache requires unsharded CP cache")
    if type(gen_num_per_cycle) is not int or gen_num_per_cycle < 0:
        raise ValueError("V4.1 gen_num_per_cycle must be a nonnegative integer")
    return {
        "csa_kv": V41PageLayout(128, 256, 32),  # FP4 + E4M3, group 16
        "hca_kv": V41PageLayout(256, 256, 32),
        # Ratio-2 producers use half the entries, but retain a full index page.
        "indexer_kv": V41PageLayout(256, 64, 4),  # FP4 + UE8M0, group 32
        "csa_state": V41PageLayout((2 + gen_num_per_cycle + 1) & ~1, 1024 * 4),
        "swa_kv": V41PageLayout(
            (128 + gen_num_per_cycle + 1) & ~1, 512, 16, 512
        ),  # FP8 + UE8M0, group 32
    }


def _validate_cache_config(text: Mapping[str, Any]) -> None:
    expected = {
        "num_hidden_layers": 40,
        "head_dim": 512,
        "index_head_dim": 128,
        "sliding_window": 128,
        "kv_source_layer_ids": [2, 8, 14, 20],
        "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
        "candidate_source_layer_id": 20,
    }
    for name, value in expected.items():
        if text.get(name) != value:
            raise ValueError(f"V4.1 exact cache requires {name}={value!r}")
    if text.get("compress_ratios", [])[:40] != [0, 0] + [2] * 18 + [1] * 20:
        raise ValueError("V4.1 exact cache requires ratios 0/2/1 at layers 0/2/20")


@dataclass(frozen=True)
class V41CacheBinding:
    source_layer_id: int
    tag: str
    storage_layer_id: int
    is_read_replica: bool = False


def dsv41_cache_bindings(
    text: Mapping[str, Any], pp_size: int = 1, pp_rank: int = 0
) -> tuple[V41CacheBinding, ...]:
    """Physical allocations; source IDs are global and storage IDs stage-local."""
    _validate_cache_config(text)
    counts = dsv41_pp_partition(40, pp_size)
    if type(pp_rank) is not int or not 0 <= pp_rank < pp_size:
        raise ValueError("V4.1 pp_rank is outside its PP group")
    begin = sum(counts[:pp_rank])
    bindings = []
    for layer in range(begin, begin + counts[pp_rank]):
        tags = ["swa_kv"]
        if layer in text["kv_source_layer_ids"]:
            tags += ["csa_kv" if layer < 20 else "hca_kv", "indexer_kv"]
            if layer < 20:
                tags.append("csa_state")
        bindings.extend(V41CacheBinding(layer, tag, layer - begin) for tag in tags)
    if begin == 21:
        # Local slot zero is storage only for these tags; L21 has no producer weights.
        bindings.extend(
            V41CacheBinding(20, tag, 0, True) for tag in ("hca_kv", "indexer_kv")
        )
    return tuple(bindings)


def dsv41_cache_binding_for_layer(
    text: Mapping[str, Any],
    layer_id: int,
    tag: str,
    pp_size: int = 1,
    pp_rank: int = 0,
) -> V41CacheBinding:
    bindings = dsv41_cache_bindings(text, pp_size, pp_rank)
    if not any(b.source_layer_id == layer_id and b.tag == "swa_kv" for b in bindings):
        raise ValueError(f"V4.1 layer {layer_id} does not belong to this stage")
    source = layer_id
    if tag in ("csa_kv", "hca_kv", "indexer_kv"):
        # Index keys follow KV ownership, not the more frequent top-k selections.
        source = max(
            (i for i in text["kv_source_layer_ids"] if i <= layer_id), default=-1
        )
    for binding in bindings:
        if binding.source_layer_id == source and binding.tag == tag:
            return binding
    raise ValueError(f"V4.1 layer {layer_id} has no cache binding for {tag}")


def build_dsv41_kv_cache_spec_descs(text: Mapping[str, Any]):
    """Global producer descriptors; native PP projection adds stage-one replicas."""
    bindings = dsv41_cache_bindings(text)
    from rtp_llm.ops import (
        CacheCapacityPolicyDesc,
        CacheCpPolicyDesc,
        CacheReusePolicyDesc,
        CpBlockMappingMode,
        CpBlockSliceMode,
        CpPrefillSliceLayout,
        DataType,
        KVCacheSpecDesc,
        KVCacheSpecType,
        OpaqueBlockEntryCountMode,
    )

    layouts = dsv41_page_layouts(0)
    result = [[] for _ in range(40)]
    for binding in bindings:
        tag = binding.tag
        layout = layouts[tag]
        desc = KVCacheSpecDesc()
        desc.tag = tag
        desc.is_state_cache = tag in ("swa_kv", "csa_state")
        if desc.is_state_cache:
            capacity = CacheCapacityPolicyDesc()
            capacity.single_block_per_request = True
            desc.capacity = capacity
            reuse = CacheReusePolicyDesc()
            reuse.enable_prefix_reuse = False
            desc.reuse = reuse
        desc.cache_type = (
            KVCacheSpecType.OPAQUE_STATE
            if desc.is_state_cache
            else KVCacheSpecType.OPAQUE_KV
        )
        desc.dtype = desc.entry_dtype = (
            DataType.TYPE_FP32 if tag == "csa_state" else DataType.TYPE_UINT8
        )
        desc.entry_elems = (
            1024
            if tag == "csa_state"
            else layout.payload_bytes_per_entry + layout.scale_bytes_per_entry
        )
        desc.entry_count_mode = (
            OpaqueBlockEntryCountMode.STATE_RING
            if desc.is_state_cache
            else OpaqueBlockEntryCountMode.KERNEL_BLOCK_COMPRESSED
        )
        desc.compression_ratio = {
            "csa_kv": 2,
            "hca_kv": 1,
            "indexer_kv": 1,
            "csa_state": 2,
            "swa_kv": 128,
        }[tag]
        desc.state_ring_include_gen_num_per_cycle = desc.is_state_cache
        if tag == "swa_kv":
            desc.block_stride_bytes_alignment = 512
        cp = CacheCpPolicyDesc()
        cp.mapping = CpBlockMappingMode.NONE
        cp.slice = CpBlockSliceMode.NONE
        cp.scale_seq_size = False
        cp.align_payload = False
        cp.prefill_slice_layout = CpPrefillSliceLayout.NONE
        desc.cp = cp
        result[binding.source_layer_id].append(desc)
    return result


def configure_dsv41_kv_cache(model_config) -> None:
    from rtp_llm.ops import HybridAttentionType

    attn = model_config.attn_config
    physical = int(attn.tokens_per_block)
    kernel = int(attn.kernel_tokens_per_block)
    if physical == 64:
        if kernel == physical:
            kernel = 256
        physical = 256
    if kernel == 0:
        kernel = physical
    dsv41_page_layouts(0, physical, kernel)
    descs = build_dsv41_kv_cache_spec_descs(model_config.deepseek_v41_config)
    attn.tokens_per_block = physical
    attn.kernel_tokens_per_block = kernel
    hybrid = model_config.hybrid_attention_config
    hybrid.hybrid_attention_types = [HybridAttentionType.NONE] * 40
    hybrid.enable_independent_kv_cache_pools = True
    model_config.kv_cache_spec_descs = descs
