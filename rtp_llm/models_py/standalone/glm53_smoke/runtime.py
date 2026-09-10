"""Build the production model and explicit, bounded hybrid cache fixtures."""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from .model import install


def build_model(checkpoint, phase, batch, seq_len):
    from rtp_llm.config.engine_config import EngineConfig
    from rtp_llm.config.py_config_modules import PyEnvConfigs
    from rtp_llm.model_factory import ModelFactory
    from rtp_llm.models_py.distributed.collective_torch import (
        init_distributed_environment,
    )
    from rtp_llm.ops import CPRotateMethod
    from rtp_llm.ops.compute_ops import init_exec_ctx

    install()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    os.environ.setdefault(
        "RTP_LLM_CPU_TP_BROADCASTER_DIR", f"/tmp/glm53-smoke-{os.getuid()}"
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = PyEnvConfigs()
    cfg.model_args.model_type = "glm5_3_flash"
    cfg.model_args.ckpt_path = checkpoint
    cfg.model_args.tokenizer_path = checkpoint
    cfg.model_args.max_seq_len = seq_len + 256
    cfg.model_args.act_type = "bf16"
    cfg.role_config.role_type = phase.upper()
    pc = cfg.parallelism_config
    pc.world_size = pc.local_world_size = 8
    pc.world_rank = pc.local_rank = pc.ep_rank = rank
    pc.tp_size = pc.ffn_tp_size = 8 if phase == "prefill" else 1
    pc.dp_size = 1 if phase == "prefill" else 8
    pc.tp_rank = pc.ffn_tp_rank = rank if phase == "prefill" else 0
    pc.dp_rank = 0 if phase == "prefill" else rank
    pc.ep_size = 8
    pc.prefill_cp_config.kv_cache_sharded = True
    pc.prefill_cp_config.prefill_cp_size = 8
    pc.prefill_cp_config.method = (
        CPRotateMethod.DISABLED if phase == "prefill" else CPRotateMethod.PREFILL_CP
    )
    cfg.concurrency_config.concurrency_limit = batch
    cfg.kv_cache_config.seq_size_per_block = 128
    cfg.kv_cache_config.kernel_seq_size_per_block = 128
    cfg.moe_config.moe_strategy = os.environ["MOE_STRATEGY"]
    cfg.moe_config.use_deepep_moe = False
    cfg.moe_config.use_deepep_low_latency = False
    cfg.py_hw_kernel_config.enable_cuda_graph = phase == "decode"
    engine = EngineConfig.create(cfg)
    model_cfg = ModelFactory.create_model_config(
        model_args=cfg.model_args,
        lora_config=cfg.lora_config,
        kv_cache_config=engine.kv_cache_config,
        profiling_debug_logging_config=engine.profiling_debug_logging_config,
        generate_env_config=cfg.generate_env_config,
        embedding_config=cfg.embedding_config,
        quantization_config=cfg.quantization_config,
        render_config=cfg.render_config,
        vit_config=cfg.vit_config,
    )
    ModelFactory.update_engine_config_from_model_config(engine, model_cfg)
    model_cfg.gen_num_per_cycle = engine.sp_config.gen_num_per_cycle
    engine.nccl_comm_config.nccl_ip = os.environ["MASTER_ADDR"]
    init_distributed_environment(
        pc, engine.nccl_comm_config, int(os.environ["MASTER_PORT"])
    )
    init_exec_ctx(
        device_id=rank,
        trace_memory=False,
        enable_comm_overlap=engine.device_resource_config.enable_comm_overlap,
        mla_ops_type=int(model_cfg.mla_ops_type),
    )
    # DeviceBase reparses serving argv. Supply the already constructed config
    # during its singleton initialization; smoke CLI flags are not server args.
    from unittest.mock import patch
    from rtp_llm.device import get_current_device

    with patch("rtp_llm.server.server_args.server_args.setup_args", return_value=cfg):
        get_current_device()
    gpt = ModelFactory._create_model(
        model_cfg, engine, cfg.vit_config, merge_lora=False
    )
    gpt.load()
    return gpt


def make_cache(model, phase, batch, seq_len):
    from rtp_llm.ops.compute_ops import CacheGroupType, KVCache, KVCacheRegionName

    prefill = phase == "prefill"
    pages = math.ceil((seq_len + 1) / 128)
    local_pages = math.ceil(pages / 8) if prefill else pages
    blocks = batch * local_pages + 1
    cache = KVCache()
    cache.seq_size_per_block = cache.kernel_seq_size_per_block = 128
    cache.num_kv_heads = 8 if prefill else 64
    cache.head_dim = 256
    cache.use_mla = True
    cache.kv_lora_rank = 512
    cache.rope_head_dim = 0
    cache.layer_group_types = [CacheGroupType.LINEAR] * 3 + [CacheGroupType.FULL]
    cache.group_region_names = [KVCacheRegionName.DEFAULT] * 2 + [
        KVCacheRegionName.INDEXER_KV,
        KVCacheRegionName.INDEXER_STATE,
    ]
    cache.group_seq_size_per_block = [128] * 4
    cache.layer_region_to_group_id = [[1] + [-1] * 7 for _ in range(3)] + [
        [0, -1, -1, 2, 3, -1, -1, -1]
    ]
    converter = model.layers[0].self_attn.decode_kda.linear_cache_converter
    linear_width = converter.block_size_bytes // 2
    bases = [
        torch.zeros((batch * 2 + 1, linear_width), dtype=torch.bfloat16, device="cuda")
        for _ in range(3)
    ]
    bases.append(torch.zeros((blocks, 128, 512), dtype=torch.bfloat16, device="cuda"))
    scales = [torch.empty(0) for _ in range(3)] + [
        torch.zeros((blocks, 128, 132), dtype=torch.uint8, device="cuda")
    ]
    regions = [[base] + [torch.empty(0) for _ in range(7)] for base in bases]
    regions[3][3] = torch.zeros((blocks, 4224), dtype=torch.uint8, device="cuda")
    regions[3][4] = torch.zeros(
        (blocks, (1 if prefill else 8), 128, 2), dtype=torch.float32, device="cuda"
    )
    scale_regions = [[scale] + [torch.empty(0) for _ in range(7)] for scale in scales]
    cache.kv_cache_base_by_layer = bases
    cache.kv_scale_base_by_layer = scales
    cache.kv_cache_base_by_layer_region = regions
    cache.kv_scale_base_by_layer_region = scale_regions
    cache.kv_cache_base_by_layer_region_flat = [
        x if x is not None else torch.empty(0) for row in regions for x in row
    ]
    model.initialize(
        SimpleNamespace(
            kv_cache=cache,
            is_speculative=False,
            is_decode_role=not prefill,
            max_context_batch_size=batch,
        )
    )
    return cache


def make_inputs(phase, batch, seq_len, seed):
    from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs, get_typemeta

    prefill = phase == "prefill"
    rows = batch * seq_len if prefill else batch
    pages = math.ceil((seq_len + 1) / 128)
    local_pages = math.ceil(pages / 8) if prefill else pages
    full = torch.zeros((batch, pages), dtype=torch.int32)
    full[:, :local_pages] = torch.arange(
        1, batch * local_pages + 1, dtype=torch.int32
    ).reshape(batch, local_pages)
    linear = (
        torch.arange(batch, dtype=torch.int32)[:, None] * 2
        + torch.arange(pages, dtype=torch.int32)[None, :] % 2
        + 1
    )
    tables = [full, linear, full.clone(), full.clone()]
    a = PyAttentionInputs()
    a.is_prefill = prefill
    a.is_cuda_graph = False
    a.input_lengths = torch.full(
        (batch,), seq_len if prefill else 1, dtype=torch.int32, device="cuda"
    )
    a.prefix_lengths = torch.zeros(
        batch if prefill else 0, dtype=torch.int32, device="cuda"
    )
    a.sequence_lengths = torch.full(
        (0 if prefill else batch,), seq_len, dtype=torch.int32, device="cuda"
    )
    a.sequence_lengths_plus_1_d = a.sequence_lengths + 1
    a.cu_seqlens = torch.arange(batch + 1, dtype=torch.int32, device="cuda") * (
        seq_len if prefill else 1
    )
    a.cu_kv_seqlens = torch.arange(batch + 1, dtype=torch.int32, device="cuda") * (
        seq_len if prefill else seq_len + 1
    )
    a.context_total_kv_length = batch * seq_len if prefill else 0
    a.total_tokens = rows
    a.padding_offset = torch.zeros(rows, dtype=torch.int32, device="cuda")
    a.kv_cache_block_id_host_by_group = tables
    a.kv_cache_kernel_block_id_host_by_group = tables
    a.kv_cache_kernel_block_id_device_by_group = [x.cuda() for x in tables]
    a.kv_cache_layer_to_group = torch.tensor([1, 1, 1, 0], dtype=torch.int32)
    a.kv_cache_block_id_host = a.kv_cache_kernel_block_id_host = tables[0]
    a.kv_cache_block_id_device = a.kv_cache_kernel_block_id_device = (
        a.kv_cache_kernel_block_id_device_by_group[0]
    )
    a.dtype = get_typemeta(torch.empty(0, dtype=torch.bfloat16))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(
        100, 30000, (rows,), dtype=torch.int32, generator=generator
    ).cuda()
    return PyModelInputs(input_ids=ids, attention_inputs=a)
