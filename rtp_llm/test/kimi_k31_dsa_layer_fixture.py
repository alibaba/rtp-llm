"""One K3.1 DSA decode layer (KimiK31DSA) built as the decode server builds it.

- Config: a K3.1 config.json through the model class (the target binds the FP8 Indexer), with the
  serving geometry of the PD smoke launcher: Page-RR pages of 1024 tokens over the TP8 = DCP8
  group, kernel pages of 128, max_seq_len 1M, Q = gen_num_per_cycle + 1 queries per request.
- Weights: random BF16 global tensors sharded as the K3.1 loader shards them; the Indexer
  projections are block-quantized to FP8 as KimiK31LoadIndexerFp8Weight stores them.
- Caches: the "full" (BF16 latent + suffix) and "indexer_kv" (E4M3 key + FP32 scale) groups have
  their own page pools and page tables, as the two tagged cache groups of the server.
- Workspace: one DsaWorkspaceOwner with the model adapter's graph capacity and V-up weights.
- CUDA Graph: GraphSlot captures one (Q, bucket) like CudaGraphRunner: static inputs of the padded
  bucket, the runner's global-width page tables, padded rows as dummy requests on reserved block 0
  with zero hidden rows, prepare_cuda_graph before every replay.

Recorder keeps the tensors the layer hands to the Indexer selection and to sparse MLA. Under a
graph they are the captured buffers, so every replay leaves its own values in them.
"""

import json
import math
import os
from types import SimpleNamespace

import torch

from rtp_llm.models.kimi_k3.kimi_k31 import KimiK31
from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_kv_workspace import (
    DsaWorkspaceOwner,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_page_rr_impl import (
    SparseMlaPageRRImpl,
)
from rtp_llm.ops import NcclCommConfig, ParallelismConfig, RoleType
from rtp_llm.ops.compute_ops import LayerKVCache, rtp_llm_ops
from rtp_llm.utils.model_weight import W

PAGE, KERNEL_PAGE = 1024, 128
INDEX_ENTRY = 128 + 4  # E4M3 key bytes, then one FP32 scale per token
TAGS = ("full", "indexer_kv")
MAX_TOKENS = 512  # largest decode batch in tokens (requests x queries)


def init_distributed(tp_size):
    # The decode role's NCCL settings on the SM103 DCP collective path (server_launcher.py).
    os.environ.setdefault("NCCL_GRAPH_REGISTER", "0")
    os.environ.setdefault("NCCL_MAX_CTAS", "8")
    world_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    parallel = ParallelismConfig()
    parallel.tp_size, parallel.dp_size = tp_size, world_size // tp_size
    parallel.ep_size = parallel.world_size = parallel.local_world_size = world_size
    parallel.world_rank = parallel.local_rank = world_rank
    parallel.ffn_sp_size = tp_size
    parallel.role_type = RoleType.DECODE
    parallel.prefill_cp_config.kv_cache_sharded = True
    nccl = NcclCommConfig()
    nccl.nccl_ip = os.environ["MASTER_ADDR"]
    collective.init_distributed_environment(
        parallel, nccl, int(os.environ["MASTER_PORT"]), disable_custom_all_reduce=True
    )
    return parallel


def load_config(path, *, max_seq_len, queries, layers=None):
    """K3.1 target config; `layers` widens a short checkpoint to the full 93-layer schedule."""
    with open(path) as reader:
        config_json = json.load(reader)
    # Kimi-K3.1-Pro-1103 omits dsa_config.indexer_use_fp8, which DsaConfig requires; the target binds
    # the FP8 Indexer whatever the checkpoint says (KimiK31._dsa_config).
    config_json["text_config"]["dsa_config"].setdefault("indexer_use_fp8", False)
    if layers is not None:
        text = config_json["text_config"]
        dsa = list(range(4, layers + 1, 4)) + ([layers] if layers % 4 else [])
        text["num_hidden_layers"] = layers
        text["linear_attn_config"]["full_attn_layers"] = dsa
        text["linear_attn_config"]["kda_layers"] = [n for n in range(1, layers + 1) if n not in dsa]
        text["dsa_config"]["dsa_layers"] = dsa
    config = KimiK31._from_config_json(config_json)
    config.init_precision_config(kv_cache_config=None, act_type="BF16")
    if not config.k3_runtime_config.dsa.indexer_use_fp8:
        raise ValueError("the K3.1 target layer must bind the FP8 Indexer")
    config.attn_config.tokens_per_block = PAGE
    config.attn_config.kernel_tokens_per_block = KERNEL_PAGE
    config.max_seq_len = max_seq_len
    config.gen_num_per_cycle = queries - 1
    return config


def _fp8_projection(source):
    """[in, out] BF16 -> the loader's E4M3 [out, in] weight and its block scales."""
    from rtp_llm.models.kimi_k3.fp8_weight import KimiK3LoadFp8Weight
    from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import is_deep_gemm_e8m0_used
    from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import _transform_scale_ue8m0

    if not is_deep_gemm_e8m0_used():
        raise ValueError("the harness reproduces the UE8M0 storage of SM100-class GPUs only")
    weight, scale = KimiK3LoadFp8Weight._quantize_matrix(source.T.contiguous(), use_ue8m0=True)
    return weight, _transform_scale_ue8m0(scale, mn=weight.shape[0])


def layer_weights(config, parallel, seed):
    """(this rank's weights of one DSA layer, global MLA up-projections for references)."""
    a, hidden = config.attn_config, config.hidden_size
    tp, rank = parallel.tp_size, parallel.tp_rank
    heads, q_dim = a.head_num, a.nope_head_dim + a.rope_head_dim
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, fan_in, dtype=torch.bfloat16):
        return (torch.randn(*shape, generator=gen, device="cuda") * fan_in**-0.5).to(dtype)

    def shard(tensor, dim):
        return tensor.chunk(tp, dim)[rank].contiguous()

    q_a = randn(hidden, a.q_lora_rank, fan_in=hidden)
    kv_a = randn(hidden, a.kv_lora_rank + a.rope_head_dim, fan_in=hidden)
    gate = randn(hidden, heads * a.v_head_dim, fan_in=hidden)
    q_b = randn(a.q_lora_rank, heads * q_dim, fan_in=a.q_lora_rank)
    output = randn(heads * a.v_head_dim, hidden, fan_in=heads * a.v_head_dim)
    key_up = randn(heads, a.nope_head_dim, a.kv_lora_rank, fan_in=a.nope_head_dim)
    value_up = randn(heads, a.kv_lora_rank, a.v_head_dim, fan_in=a.kv_lora_rank)
    index_heads, index_dim = a.indexer_head_num, a.indexer_head_dim
    index_q, index_q_scale = _fp8_projection(
        randn(a.q_lora_rank, index_heads * index_dim, fan_in=a.q_lora_rank)
    )
    index_k, index_k_scale = _fp8_projection(randn(hidden, index_dim, fan_in=hidden))
    weights = {
        W.mla_fusedqkrope_w: torch.cat((q_a, kv_a, shard(gate, 1)), dim=1),
        W.mla_q_b_w: shard(q_b, 1),
        W.attn_o_w: shard(output, 0),
        W.mla_q_a_ln_gamma: torch.ones(a.q_lora_rank, dtype=torch.bfloat16, device="cuda"),
        W.mla_kv_a_ln_gamma: torch.ones(a.kv_lora_rank, dtype=torch.bfloat16, device="cuda"),
        W.mla_kc: shard(key_up, 0),
        W.mla_vc: shard(value_up, 0),
        W.mla_indexer_qb_w: index_q,
        W.mla_indexer_qb_s: index_q_scale,
        W.mla_indexer_k_w: index_k,
        W.mla_indexer_k_s: index_k_scale,
        W.mla_indexer_k_norm_w: torch.ones(index_dim, dtype=torch.bfloat16, device="cuda"),
        W.mla_indexer_k_norm_b: torch.zeros(index_dim, dtype=torch.bfloat16, device="cuda"),
        W.mla_indexer_weights_proj_w: randn(hidden, index_heads, fan_in=hidden, dtype=torch.float32),
    }
    return weights, SimpleNamespace(key_up=key_up, value_up=value_up)


def build_layer(config, parallel, weights, layer_index):
    from rtp_llm.models_py.modules.kimi_k3.dsa import KimiK31DSA

    layer = KimiK31DSA(config, parallel, weights, layer_index)
    if type(layer.indexer.wq_b).__name__ != "CudaFp8GEMMLinear":
        raise ValueError(f"Indexer projection is {type(layer.indexer.wq_b).__name__}, not FP8 GEMM")
    return layer


def build_owner(config, parallel, layers, *, max_requests, queries):
    """The model adapter's workspace owner (KimiK31ModelAdapter._setup_dsa)."""
    from rtp_llm.models_py.utils.flash_mla_policy import configure_flash_mla_l2_persist

    configure_flash_mla_l2_persist()
    owner = DsaWorkspaceOwner.from_model(config, parallel)
    owner.graph_max_requests = max_requests
    owner.graph_max_queries = queries
    owner.graph_cache_groups = len(TAGS)
    owner.bind_value_weights(layers)
    return owner


# Page-RR geometry: global page g lives on rank g % tp at local page g // tp.


def local_tokens(length, tp, rank):
    """How many of a request's first `length` positions this rank owns."""
    pages, tail = divmod(length, PAGE)
    full = pages // tp + (1 if rank < pages % tp else 0)
    return full * PAGE + (tail if pages % tp == rank else 0)


def owner_columns(length, tp, rank):
    return math.ceil(local_tokens(length, tp, rank) / KERNEL_PAGE)


def local_to_global(local, tp, rank):
    return ((local // PAGE) * tp + rank) * PAGE + local % PAGE


def global_owner(position, tp):
    """(owner rank, owner-local token) of global positions."""
    page = position // PAGE
    return page % tp, (page // tp) * PAGE + position % PAGE


def graph_columns(max_seq_len, queries):
    """The runner's global-width kernel page table (captureKernelBlockTableWidth, full group)."""
    return math.ceil((max_seq_len + queries) / PAGE) * (PAGE // KERNEL_PAGE)


def sp_unit(tp, queries):
    """Requests are padded so every rank holds the same number of token rows."""
    return tp // math.gcd(tp, queries)


class Pool:
    """One cache group's pages on this rank; page 0 stays reserved for dummy requests."""

    def __init__(self, tag, pages, device):
        self.tag = tag
        if tag == "full":
            self.cache = torch.zeros(pages, KERNEL_PAGE, 512 + 64, dtype=torch.bfloat16, device=device)
        else:
            self.cache = torch.zeros(pages, KERNEL_PAGE, INDEX_ENTRY, dtype=torch.uint8, device=device)
        self.free = list(range(pages - 1, 0, -1))

    def allocate(self, count, generator):
        if count > len(self.free):
            return None
        order = torch.randperm(len(self.free), generator=generator)[:count].tolist()
        chosen = [self.free[i] for i in order]
        taken = set(order)
        self.free = [page for i, page in enumerate(self.free) if i not in taken]
        return torch.tensor(chosen, dtype=torch.int32, device=self.cache.device)

    def release(self, pages):
        self.free.extend(pages.tolist())

    def layer_cache(self, layer_index):
        return LayerKVCache(self.cache, PAGE, layer_index, self.tag)

    def slots(self, pages, local):
        """Slots of owner-local tokens of a request whose owner pages are `pages`."""
        return pages.long()[local // KERNEL_PAGE] * KERNEL_PAGE + local % KERNEL_PAGE

    def checksum(self, chunk=1024):
        """Per-page fingerprint: sums of the page's 32-bit words, plain and position-weighted."""
        words = self.cache.view(self.cache.shape[0], -1).view(torch.int32)
        weights = torch.arange(1, words.shape[1] + 1, device=words.device)
        result = torch.empty(words.shape[0], 2, dtype=torch.int64, device=words.device)
        for first in range(0, words.shape[0], chunk):
            block = words[first : first + chunk].long()
            result[first : first + chunk, 0] = block.sum(1)
            result[first : first + chunk, 1] = (block * weights).sum(1)
        return result


class Request:
    """A decode request on this rank: random history of `prefix` positions, pages for `capacity`."""

    def __init__(self, seed, prefix, capacity):
        self.seed, self.prefix, self.capacity = seed, prefix, capacity
        self.pages = {}

    def allocate(self, pools, tp, rank, generator):
        columns = owner_columns(self.capacity, tp, rank)
        for tag in TAGS:
            pages = pools[tag].allocate(columns, generator)
            if pages is None:
                self.release(pools)
                return False
            self.pages[tag] = pages
        return True

    def release(self, pools):
        for tag, pages in self.pages.items():
            pools[tag].release(pages)
        self.pages = {}

    def write_history(self, pools, tp, rank, start=0, chunk=1 << 18):
        """Random history at positions [start, prefix) through the production writers."""
        begin, end = local_tokens(start, tp, rank), local_tokens(self.prefix, tp, rank)
        gen = torch.Generator(device="cuda").manual_seed(self.seed * 8191 + rank)
        for first in range(begin, end, chunk):
            local = torch.arange(first, min(first + chunk, end), device="cuda")
            latent = torch.randn(local.numel(), 576, generator=gen, device="cuda").bfloat16()
            pools["full"].cache.view(-1, 576)[pools["full"].slots(self.pages["full"], local)] = latent
            keys = torch.randn(local.numel(), 128, generator=gen, device="cuda").bfloat16()
            rtp_llm_ops.indexer_k_quant_and_cache(
                keys, pools["indexer_kv"].cache, pools["indexer_kv"].slots(self.pages["indexer_kv"], local),
                128, "ue8m0",
            )


def attention_inputs(requests, queries, padded, *, graph_width, tp, rank, device, capture_len=None):
    """One decode step's attention inputs per tag, padded to `padded` requests as the runner pads.

    graph_width: page-table width of a captured graph, or None for the eager owner width.
    capture_len: capture-time inputs of CudaGraphRunner (no live request, every row at this
    length on block 0, every token row masked).
    """
    batch = len(requests)
    rows, padded_rows = batch * queries, padded * queries
    prefixes = [r.prefix for r in requests] + [0] * (padded - batch)
    if capture_len is not None:
        prefixes = [capture_len - queries] * padded
    lengths = [queries] * padded
    width = graph_width or max(
        1, max(owner_columns(r.prefix + queries, tp, r_rank) for r in requests for r_rank in range(tp))
    )
    common = dict(
        is_prefill=False,
        is_target_verify=queries > 1,
        is_mtp_draft_update=False,
        is_cuda_graph=graph_width is not None,
        input_lengths=torch.tensor(lengths, dtype=torch.int32),
        prefix_lengths=torch.tensor(prefixes, dtype=torch.int32),
        input_lengths_device=torch.tensor(lengths, dtype=torch.int32, device=device),
        prefix_lengths_device=torch.tensor(prefixes, dtype=torch.int32, device=device),
        # Live: prefix + Q; padded rows as CudaGraphRunner fills them (Q for verify, 1 for decode).
        sequence_lengths_plus_1_device=torch.tensor(
            [p + queries for p in prefixes[:batch]] + [queries] * (padded - batch)
            if capture_len is None else [capture_len] * padded,
            dtype=torch.int32, device=device,
        ),
        cu_seqlens_device=torch.arange(0, padded_rows + 1, queries, dtype=torch.int32, device=device),
        padding_offset=torch.zeros(padded_rows, dtype=torch.int32, device=device),
        logical_token_count=rows if capture_len is None else padded_rows,
        physical_token_count=padded_rows,
        logical_request_count=batch,
        physical_request_count=padded,
        valid_token_mask=torch.arange(padded_rows, device=device) < rows,
        cache_store_inputs=None,
        cache_store_writer=None,
    )
    per_tag = {}
    for tag in TAGS:
        table = torch.zeros(padded, width, dtype=torch.int32, device=device)
        for index, request in enumerate(requests):
            pages = request.pages[tag][:width]
            table[index, : pages.numel()] = pages
        per_tag[tag] = SimpleNamespace(**common, kv_cache_kernel_block_id_device=table)
    return per_tag


class Recorder:
    """Tensors at the layer's two inner boundaries, refreshed by every eager call or capture."""

    def __init__(self, layer):
        self.last = {}
        indexer = layer.indexer
        quantize, select = indexer._quantize_q_k, indexer._compute_topk

        def record_quantize(query, key, *args, **kwargs):
            q_fp8, q_scale = quantize(query, key, *args, **kwargs)
            self.last["index_key"] = key
            return q_fp8, q_scale

        def record_select(q_fp8, weights, *args, **kwargs):
            selected = select(q_fp8, weights, *args, **kwargs)
            self.last.update(index_query=q_fp8, index_weights=weights, selected=selected)
            return selected

        indexer._quantize_q_k, indexer._compute_topk = record_quantize, record_select

    def attach(self, sfa):
        forward = sfa.forward

        def record_forward(q, compressed_kv, k_pe, *args, **kwargs):
            output = forward(q, compressed_kv, k_pe, *args, **kwargs)
            self.last.update(sfa_query=q, latent=compressed_kv, suffix=k_pe, sfa_output=output)
            return output

        sfa.forward = record_forward


def build_fmha(config, parallel, weights_by_layer, inputs, owner, recorder, *, graph):
    weights = SimpleNamespace(weights=weights_by_layer)
    fmha = {
        tag: SparseMlaPageRRImpl(config, parallel, weights, inputs[tag], None, graph, workspace_owner=owner)
        for tag in TAGS
    }
    recorder.attach(fmha["full"])
    return fmha


class GraphSlot:
    """One captured (Q, bucket) of consecutive DSA layers sharing the step's attention impls."""

    def __init__(self, config, parallel, layers, weights_by_layer, owner, recorder, pools, *,
                 queries, bucket, max_seq_len, pool_handle):
        tp, rank = parallel.tp_size, parallel.tp_rank
        unit = sp_unit(tp, queries)
        self.queries, self.bucket = queries, bucket
        self.padded = math.ceil(bucket / unit) * unit
        self.layers, self.recorder, self.pools = layers, recorder, pools
        self.tp, self.rank = tp, rank
        self.width = graph_columns(max_seq_len, queries)
        # Plain decode captures sequence_lengths = max_seq_len - 2, verify prefix = max_seq_len - Q.
        self.inputs = attention_inputs(
            [], queries, self.padded, graph_width=self.width, tp=tp, rank=rank, device="cuda",
            capture_len=max_seq_len - (1 if queries == 1 else 0),
        )
        self.fmha = build_fmha(config, parallel, weights_by_layer, self.inputs, owner, recorder, graph=True)
        self.caches = {
            layer.layer_idx: {tag: pools[tag].layer_cache(layer.layer_idx) for tag in TAGS} for layer in layers
        }
        rows = self.padded * queries // tp
        self.hidden = torch.zeros(rows, config.hidden_size, dtype=torch.bfloat16, device="cuda")
        self.forward()  # eager warm-up
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool_handle):
            self.output = self.forward()
        self.recorded = dict(recorder.last)
        torch.cuda.synchronize()

    def forward(self):
        primary = self.inputs["full"]
        return [layer(self.hidden, self.fmha, self.caches[layer.layer_idx], primary, None) for layer in self.layers]

    def load(self, requests, hidden):
        """Copy one step's live inputs into the captured buffers (CudaGraphRunner.prepareInputs)."""
        live = attention_inputs(
            requests, self.queries, self.padded, graph_width=self.width,
            tp=self.tp, rank=self.rank, device="cuda",
        )
        for tag in TAGS:
            target, source = self.inputs[tag], live[tag]
            for name, value in vars(source).items():
                current = getattr(target, name)
                if isinstance(current, torch.Tensor):
                    current.copy_(value)
                else:
                    setattr(target, name, value)
        self.hidden.copy_(hidden)
        for tag in TAGS:
            self.fmha[tag].prepare_cuda_graph(self.inputs[tag])

    def replay(self):
        self.graph.replay()
        return self.output
