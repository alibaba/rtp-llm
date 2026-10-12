"""Upstream V4.1 text worker on the base paged-cache runtime.

Adapted from e4a9162f6f. Serving registration remains gated while this worker
is qualified incrementally; eager short PP1/CP1/TP1 prefill admits DP1/EP1
or a single eight-rank DP8/EP8 NCCL group. Single-token decode uses the same
topologies with uniform prefill/decode phase agreement across DP peers.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from rtp_llm.config.dsv41_config import (
    v41_mhc_collapse_mode,
    v41_mhc_post_mode,
    v41_rotary_compute_device,
    v41_sparse_attention_mode,
)
from rtp_llm.models_py.model_desc.deepseek_v4_model import DeepSeekV4Model
from rtp_llm.models_py.model_desc.v41_errors import V41InputRejected
from rtp_llm.models_py.modules.dsv4.engram import Engram, EngramLayout, NgramHashState
from rtp_llm.models_py.modules.dsv4.fp8.attention import bind_attn_cache
from rtp_llm.models_py.modules.dsv4.fp8.prefill_meta import (
    clear_prefill_meta_shared_fp8,
)
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
    build_block_tables_batched,
    primary_attention_inputs,
)
from rtp_llm.models_py.modules.dsv4.utils import V41MXFP8Linear
from rtp_llm.models_py.modules.dsv4.v41_engram_host import _locked, _PinnedSharedTable
from rtp_llm.models_py.modules.dsv4.v41_engram_weights import (
    load_engram_weights,
    plan_stage_engram_sources,
)


class _SharedTableEmbedding:
    """Use upstream lookup arithmetic with the qualified explicit table lease."""

    def __init__(self, table):
        self._table = table
        self._num_sms = torch.cuda.get_device_properties(
            table._device
        ).multi_processor_count

    @property
    def weight(self):
        self._table._check_open()
        return self._table._weight.view(torch.float8_e4m3fn)

    def __call__(self, indices, device):
        from rtp_llm.models_py.modules.dsv4._engram_triton import lookup_host_rows

        table = self._table
        with table._mutex:
            table._check_open()
            if indices.device != table._device or torch.device(device) != table._device:
                raise ValueError("Engram lookup must use its registered worker device")
            return lookup_host_rows(
                table._cuda_weight, table._cuda_scales, indices, self._num_sms
            )


class DeepSeekV41Model(DeepSeekV4Model):
    def __init__(self, model_config, *args, **kwargs):
        super().__init__(model_config, *args, **kwargs)
        config = dict(getattr(model_config, "deepseek_v41_config", None) or {})
        if not config:
            raise ValueError("DeepSeek V4.1 requires deepseek_v41_config")
        self._rotary_compute_device = v41_rotary_compute_device(config)
        self._sparse_attention_mode = v41_sparse_attention_mode(config)
        self._mhc_collapse_mode = v41_mhc_collapse_mode(config)
        self._mhc_post_mode = v41_mhc_post_mode(config)
        self._v4_args.v41_config = config
        self._v4_args.n_hash_layers = 0
        self._engram_hash_state = None
        self._engram_layers = ()
        self._engram_tables = {}
        self._engram_sources = {}
        self._engram_cache_dir = Path(
            os.environ.get("DSV41_ENGRAM_CACHE_DIR", "/dev/shm")
        ).resolve()
        self._worker_lock = threading.RLock()
        self._worker_pid = os.getpid()
        self._worker_state = "new"
        self._worker_stream = None
        self._dp_group = None
        self._stage_agreement = None
        self._native_pp_owner = object()
        self._native_pp_contract = None
        self._native_pp_receiver = None

    def _check_rotary_config(self):
        if (
            v41_rotary_compute_device(self._v4_args.v41_config)
            != self._rotary_compute_device
        ):
            raise ValueError("V4.1 rotary origin changed; replace the worker")
        if self.v4 is not None:
            for layer in self.v4.layers:
                layer.attn._check_rotary_config()
                if layer.attn._rotary_compute_device != self._rotary_compute_device:
                    raise ValueError(
                        "V4.1 worker and attention rotary origins disagree"
                    )

    def _check_sparse_config(self):
        if (
            v41_sparse_attention_mode(self._v4_args.v41_config)
            != self._sparse_attention_mode
        ):
            raise ValueError("V4.1 sparse attention mode changed; replace the worker")
        if self._sparse_attention_mode is not None and self._v4_args.pp_size != 1:
            raise NotImplementedError("V4.1 short sparse worker requires PP1/CP1")
        if self.v4 is not None:
            for layer in self.v4.layers:
                layer.attn._check_sparse_config()
                if layer.attn._sparse_attention_mode != self._sparse_attention_mode:
                    raise ValueError("V4.1 worker and attention sparse modes disagree")

    def _check_collapse_config(self):
        if v41_mhc_collapse_mode(self._v4_args.v41_config) != self._mhc_collapse_mode:
            raise ValueError("V4.1 mHC collapse mode changed; replace the worker")
        if self.v4 is not None:
            for layer in self.v4.layers:
                for unit in (layer.attn_hc, layer.ffn_hc):
                    if unit.collapse_mode != self._mhc_collapse_mode:
                        raise ValueError("V4.1 worker and mHC collapse modes disagree")

    def _check_post_config(self):
        if v41_mhc_post_mode(self._v4_args.v41_config) != self._mhc_post_mode:
            raise ValueError("V4.1 mHC post mode changed; replace the worker")
        if self.v4 is not None:
            for layer in self.v4.layers:
                for unit in (layer.attn_hc, layer.ffn_hc):
                    if unit.post_mode != self._mhc_post_mode:
                        raise ValueError("V4.1 worker and mHC post modes disagree")

    def _check_execution_config(self):
        self._check_rotary_config()
        self._check_sparse_config()
        self._check_collapse_config()
        self._check_post_config()

    def _configure_dp_group(self):
        args = self._v4_args
        if args.pp_size == 2:
            from rtp_llm.models_py.distributed import collective_torch
            from rtp_llm.models_py.distributed.ep_stage_context import (
                BACKEND_FORK_NCCL_MXFP8,
                EpStageContext,
                resolve_pp_ep_opt_in,
            )
            from rtp_llm.models_py.distributed.rank_layout import Group
            from rtp_llm.models_py.modules.dsv4.transformer import (
                _v41_prefill_stage_bindings,
            )
            from rtp_llm.ops import RoleType

            pcfg = self.parallelism_config
            enabled, backend = resolve_pp_ep_opt_in(pcfg)
            if (
                (
                    pcfg.pp_size,
                    pcfg.dp_size,
                    pcfg.tp_size,
                    pcfg.ep_size,
                    pcfg.world_size,
                )
                != (2, 1, 4, 4, 8)
                or (args.dp_size, args.ep_size, args.world_size) != (1, 4, 8)
                or pcfg.role_type != RoleType.PREFILL
                or not pcfg.local_cp_enabled()
                or not enabled
                or backend != BACKEND_FORK_NCCL_MXFP8
                or not pcfg.pp_ep_enabled
                or pcfg.pp_ep_backend != backend
                or list(pcfg.pp_stage_layer_counts) != [21, 19]
                or args.pp_rank != pcfg.pp_rank
                or args.world_rank != pcfg.world_rank
                or args.ep_rank != pcfg.ep_rank
                or pcfg.tp_rank != pcfg.ep_rank
            ):
                raise NotImplementedError(
                    "V4.1 PP workers require the exact opted-in CEP4PP2 PREFILL topology"
                )
            _v41_prefill_stage_bindings(args)
            context = EpStageContext.build(pcfg, backend=backend)
            dist = torch.distributed
            if (
                dist.get_backend(context.process_group) != "nccl"
                or dist.get_world_size() != 8
                or args.ep_rank != context.group_rank
                or args.dp_rank != 0
            ):
                raise ValueError(
                    "V4.1 CEP4PP2 requires its registered NCCL STAGE roster"
                )
            if self._dp_group is not None and (
                self._dp_group is not context.process_group
                or self._stage_agreement != context
            ):
                raise RuntimeError("V4.1 stage agreement changed; replace the worker")
            if context.process_group is not collective_torch._get_group(Group.STAGE):
                raise ValueError(
                    "V4.1 stage agreement requires the registered STAGE group"
                )
            self._dp_group = context.process_group
            self._stage_agreement = context
            return
        if (args.dp_size, args.ep_size, args.world_size) == (1, 1, 1):
            return
        if (args.dp_size, args.ep_size, args.world_size) != (8, 8, 8):
            raise NotImplementedError(
                "V4.1 worker requires PP1/CP1/TP1 with DP1/EP1 or DP8/EP8"
            )
        dist = torch.distributed
        if (
            not dist.is_initialized()
            or dist.get_backend() != "nccl"
            or dist.get_world_size() != 8
            or not args.dp_rank == args.ep_rank == args.world_rank == dist.get_rank()
        ):
            raise ValueError("V4.1 DP8 worker requires its exact NCCL WORLD roster")
        if self._dp_group is not None and self._dp_group is not dist.group.WORLD:
            raise RuntimeError("V4.1 DP group changed; replace the worker")
        self._dp_group = dist.group.WORLD

    def _agreement_config(self):
        # dataclasses.asdict deep-copies ProcessGroup, which cannot be pickled.
        # Bind its verified immutable metadata while excluding peer-local ranks.
        config = {
            field.name: getattr(self._v4_args, field.name)
            for field in fields(self._v4_args)
            if not field.name.endswith("_rank") and field.name != "moe_stage_context"
        }
        context = self._v4_args.moe_stage_context
        if context is not None:
            if self._stage_agreement is None or context != self._stage_agreement:
                raise RuntimeError(
                    "V4.1 expert context disagrees with worker agreement"
                )
            config["moe_stage_context"] = {
                "world_ranks": context.world_ranks,
                "group_size": context.group_size,
                "pp_rank": context.pp_rank,
                "generation": context.generation,
                "backend": context.backend,
            }
        if self._stage_agreement is not None:
            from rtp_llm.models_py.distributed import collective_torch
            from rtp_llm.models_py.distributed.rank_layout import Group

            if self._dp_group is not collective_torch._get_group(Group.STAGE):
                raise RuntimeError(
                    "V4.1 registered stage group changed; replace the worker"
                )
        return config

    def _agree_dp(self, error, phase, device, *, native_dispatch=False):
        recoverable = isinstance(error, (ValueError, NotImplementedError))
        status = (
            1 if error is None else (0 if recoverable or not native_dispatch else 2)
        )
        if self._dp_group is not None:
            # Bind the whole model schedule before any rank enters layer
            # collectives. Request lengths/IDs remain rank-local.
            config = self._agreement_config()
            options = {
                key: os.environ.get(key)
                for key in (
                    "DSV4_MOE_CHUNK_PREFILL",
                    "DSV4_CHUNK_TOKENS",
                    "DSV4_MOE_CHUNK_TOKENS",
                    "DSV4_MOE_EXTENT_COUNT_FUSE",
                    "DSV4_MOE_FORWARD_COUNT_PLAN",
                )
            }
            digest = hashlib.sha256(
                json.dumps(
                    (phase, config, options, str(self._engram_cache_dir)),
                    sort_keys=True,
                ).encode()
            ).digest()
            signature = torch.tensor(
                [status, *digest], dtype=torch.uint8, device=device
            ).view(1, 33)
            group_size = (
                self._stage_agreement.group_size
                if self._stage_agreement is not None
                else 8
            )
            peers = torch.empty((group_size, 33), dtype=torch.uint8, device=device)
            try:
                torch.distributed.all_gather_into_tensor(
                    peers, signature, group=self._dp_group
                )
                peers = peers.cpu()
            except BaseException:
                self._worker_state = "failed"
                raise
            rejected = not bool((peers[:, 0] == 1).all())
            same_schedule = bool((peers[:, 1:] == peers[0, 1:]).all())
            if native_dispatch:
                if bool((peers[:, 0] == 2).any()):
                    self._worker_state = "failed"
                    raise RuntimeError(
                        "V4.1 native preflight failed on a DP peer"
                    ) from error
                if rejected:
                    raise V41InputRejected(
                        str(error)
                        if error is not None
                        else "V4.1 DP peer rejected request"
                    ) from error
                if not same_schedule:
                    self._worker_state = "failed"
                    raise RuntimeError("V4.1 native DP model schedules disagree")
            if rejected or not same_schedule:
                if error is not None:
                    raise error
                raise RuntimeError("V4.1 DP peer rejected request or model schedule")
        if error is not None:
            if native_dispatch:
                if recoverable:
                    raise V41InputRejected(str(error)) from error
                self._worker_state = "failed"
            raise error

    def _check_worker(self):
        if os.getpid() != self._worker_pid:
            raise RuntimeError("V4.1 workers must be spawned, not inherited after fork")
        if self._worker_state in ("failed", "closing", "closed"):
            raise RuntimeError(f"V4.1 worker is {self._worker_state}")

    def _pp_ep_grouped_kernel_available(self) -> bool:
        # V4.1's group-32 recipe consumes DeepGEMM, independently of V4's
        # FlashInfer group-32 scale interleave/graph provider.
        from importlib.metadata import PackageNotFoundError, version

        from packaging.version import Version

        try:
            import deep_gemm

            if Version(version("deep-gemm")) < Version("2.6.0"):
                return False
        except (ImportError, OSError, RuntimeError, PackageNotFoundError):
            return False
        return all(
            callable(getattr(deep_gemm, name, None))
            for name in (
                "m_grouped_fp8_fp4_gemm_nt_contiguous",
                "m_grouped_fp8_fp4_gemm_nt_masked",
            )
        )

    def _plan_engram(self):
        config = self._v4_args.v41_config
        owned = set(self.pp_layer_ids())
        if not owned.intersection(config.get("engram_layer_ids", ())):
            return {}
        # This is a metadata identity, not a hash of the checkpoint payload.
        # Every admitted span additionally binds file identity and header hash.
        identity = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        sources = plan_stage_engram_sources(
            SimpleNamespace(text=config), config["checkpoint_path"], owned, identity
        )
        total = sum(source.table.nbytes for source in sources.values())
        budget = int(os.environ.get("DSV41_ENGRAM_HOST_BYTES", "0"))
        if total > budget:
            raise ValueError(
                f"Engram tables need {total} host bytes; DSV41_ENGRAM_HOST_BYTES={budget}"
            )
        return sources

    def _check_engram_capacity(self):
        if not self._engram_sources:
            return
        directory = self._engram_cache_dir
        if not directory.is_dir():
            raise ValueError(f"Engram cache directory does not exist: {directory}")
        needed = 0
        for source in self._engram_sources.values():
            source.check_sources()
            table = source.table
            path = directory / f"rtp_llm_engram_{os.getuid()}_{table.key}"
            ready = Path(str(path) + ".json")
            if not (
                path.exists()
                and path.stat().st_size == table.nbytes
                and ready.exists()
                and ready.read_bytes() == table.manifest
            ):
                needed += table.nbytes
        if needed > shutil.disk_usage(directory).free:
            raise RuntimeError(
                f"Engram combined tables need {needed} additional bytes in {directory}"
            )

    def _initialize_impl(self, init_resource):
        with self._worker_lock:
            self._check_worker()
            args = self._v4_args
            stage_prefill = args.pp_size == 2
            if (
                args.pp_size not in (1, 2)
                or args.tp_size != 1
                or (self.parallelism_config.local_cp_enabled() and not stage_prefill)
                or self.parallelism_config.enable_sp
                or self.parallelism_config.get_ffn_tp_size() != 1
                or self.parallelism_config.ffn_sp_size != 1
                or self.parallelism_config.use_ub_comm
                or not args.fp8_kv_cache
                or args.commit_only
                or init_resource.is_speculative
                or init_resource.is_decode_role
                or self._gen_num_per_cycle not in (0, 1)
            ):
                raise NotImplementedError(
                    "V4.1 worker requires eager PP1/CP1/TP1 prefill/decode or CEP4PP2 target prefill"
                )
            self._configure_dp_group()
            if self._materialized:
                self._check_execution_config()
                return super()._initialize_impl(init_resource)
            device = next(iter(self.weight.global_weights.values())).device
            if device.type != "cuda" or torch.cuda.get_device_capability(device) != (
                12,
                0,
            ):
                raise NotImplementedError(
                    "V4.1 worker currently requires SM120 weights"
                )
            error = None
            try:
                self._check_execution_config()
                self._engram_sources = self._plan_engram()
                self._check_engram_capacity()
            except Exception as exc:
                error = exc
            self._agree_dp(error, "initialize-preflight", device)
            try:
                error = None
                try:
                    result = super()._initialize_impl(init_resource)
                    self._check_execution_config()
                except Exception as exc:
                    error = exc
                self._agree_dp(error, "initialize-ready", device)
            except BaseException:
                self._worker_state = "failed"
                self.close()
                raise
            self._worker_state = "ready"
            self._worker_stream = torch.cuda.current_stream(self._weight_device())
            return result

    def _load_extra_weights(self, weights):
        if not self._engram_sources:
            return
        config = self._v4_args.v41_config
        layout = EngramLayout(config)
        device = self._weight_device()
        hash_state = NgramHashState(layout, config["checkpoint_path"], device=device)
        if hash_state.token_map.numel() != self._v4_args.vocab_size:
            raise ValueError("Engram tokenizer size disagrees with model vocabulary")
        by_global_id = {layer.layer_id: layer for layer in self.v4.layers}
        modules = []
        # Reserve and lease all tables under one admission lock. Registration
        # may then overlap across workers without racing capacity or deletion.
        lock = Path("/tmp") / f"rtp_llm_v41_engram_admission_{os.getuid()}.lock"
        with _locked(lock):
            self._check_engram_capacity()
            for layer_id, source in self._engram_sources.items():
                table = _PinnedSharedTable(
                    source.table,
                    max_bytes=source.table.nbytes,
                    directory=self._engram_cache_dir,
                )
                self._engram_tables[layer_id] = table
        for layer_id, source in self._engram_sources.items():
            table = self._engram_tables[layer_id]
            table.register_cuda(device)
            tensors = load_engram_weights(
                source, max_bytes=source.weight_bytes, device=device
            )
            projection = V41MXFP8Linear(
                tensors["weight"], tensors["scales"].view(torch.float8_e8m0fnu)
            )
            index = layout.layer_ids.index(layer_id)
            module = Engram(
                layout,
                index,
                _SharedTableEmbedding(table),
                projection,
                tensors["q_weight"],
                tensors["k_weight"],
                source.norm_eps,
            )
            modules.append((by_global_id[layer_id], index, module))
        # Publish only after every owned table and projection is ready.
        for layer, _, module in modules:
            layer.engram = module
        self._engram_layers = tuple((layer, index) for layer, index, _ in modules)
        self._engram_hash_state = hash_state

    def _prepare_engram(self, inputs):
        if not self._engram_layers:
            return
        windows = inputs.engram_token_windows
        if (
            windows is None
            or windows.ndim != 2
            or windows.shape != (inputs.input_ids.numel(), 4)
            or windows.dtype != torch.int32
            or not windows.is_contiguous()
        ):
            raise ValueError(
                "Engram requires contiguous int32 [tokens, 4] token windows"
            )
        windows = windows.to(device=self._weight_device(), non_blocking=True)
        ids = inputs.input_ids.reshape(-1).to(windows.device)
        if bool(((windows < -1) | (windows >= self._v4_args.vocab_size)).any()):
            raise ValueError("Engram token history contains an invalid token ID")
        if bool(((windows[:, 0] >= 0) & (windows[:, 0] != ids)).any()):
            raise ValueError("Engram current tokens disagree with input_ids")
        hashes = self._engram_hash_state.hash_token_windows(windows)
        token_mask = windows[:, 0] >= 0
        for layer, index in self._engram_layers:
            layer.engram_hashes = hashes[:, index].contiguous()
            layer.engram_token_mask = token_mask

    def _clear_forward_state(self):
        if self.v4 is None:
            return
        for layer in self.v4.layers:
            layer.engram_hashes = None
            layer.engram_token_mask = None
            layer.attn_hc.pre_mix_out = None
            layer.ffn_hc.pre_mix_out = None
        clear_prefill_meta_shared_fp8(self.v4)

    def _validate_prefill(self, inputs, attn):
        lengths, prefixes = attn.input_lengths, attn.prefix_lengths
        idle = self._dp_group is not None and inputs.input_ids.numel() == 0
        if (
            lengths is None
            or prefixes is None
            or lengths.ndim != 1
            or prefixes.shape != lengths.shape
            or not (0 if idle else 1)
            <= lengths.numel()
            <= self._max_generate_batch_size
            or lengths.dtype not in (torch.int32, torch.int64)
            or prefixes.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("V4.1 prefill requires matching integer request lengths")
        lengths, prefixes = lengths.cpu(), prefixes.cpu()
        if (
            self._sparse_attention_mode is not None
            and not idle
            and (
                lengths.numel() != 1 or bool(((lengths > 128) | (prefixes != 0)).any())
            )
        ):
            raise NotImplementedError(
                "V4.1 short sparse worker requires one initial prefill"
            )
        if idle and lengths.numel():
            raise ValueError("V4.1 idle DP rank must have no requests")
        if (
            bool((lengths <= 0).any())
            or bool((prefixes < 0).any())
            or int(lengths.sum()) != inputs.input_ids.numel()
        ):
            raise ValueError("V4.1 prefill lengths must cover nonempty requests")
        if bool(((prefixes + lengths) > min(512, self._v4_args.max_seq_len)).any()):
            raise NotImplementedError(
                "V4.1 prefill requires positions within the qualified 512-token context"
            )
        cu = attn.cu_seqlens
        expected_cu = torch.cat((torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)))
        if (
            cu is not None
            and cu.numel()
            and (
                cu.dtype not in (torch.int32, torch.int64)
                or not torch.equal(cu.cpu(), expected_cu)
            )
        ):
            raise ValueError("V4.1 prefill cumulative lengths disagree with requests")
        positions = attn.combo_position_ids
        if positions is not None and positions.numel():
            expected = torch.cat(
                [
                    torch.arange(p, p + n)
                    for p, n in zip(prefixes.tolist(), lengths.tolist())
                ]
            )
            if positions.dtype not in (torch.int32, torch.int64) or not torch.equal(
                positions.cpu(), expected
            ):
                raise ValueError("V4.1 prefill positions disagree with requests")

    def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
        if is_cuda_graph:
            raise NotImplementedError("V4.1 worker CUDA graphs are not integrated")
        return None

    @torch.inference_mode()
    def prepare_native_pp_prefill(self, inputs):
        """Agree on native CP admission; PP execution remains a separate gate."""
        from rtp_llm.models_py.modules.dsv4.native_pp_prefill import (
            prepare_native_pp_prefill_dispatch,
        )

        with self._worker_lock:
            self._check_worker()
            if self._worker_state != "ready":
                raise RuntimeError("V4.1 worker is not initialized")
            if self._v4_args.pp_size != 2:
                raise NotImplementedError("V4.1 native PP preflight requires CEP4PP2")
            dispatch = error = None
            try:
                self._check_execution_config()
                dispatch = prepare_native_pp_prefill_dispatch(self, inputs)
            except Exception as exc:
                error = exc
            try:
                with torch.cuda.stream(self._worker_stream):
                    self._agree_dp(
                        error,
                        (
                            "native-pp-prefill-preflight",
                            None if dispatch is None else dispatch.schedule,
                        ),
                        self._weight_device(),
                        native_dispatch=True,
                    )
                return dispatch
            finally:
                # A validator traceback references this frame and its native
                # inputs. Break the saved-exception cycle so rejected leases
                # release during unwinding rather than waiting for cyclic GC.
                error = None

    @torch.inference_mode()
    def _configure_native_pp_engine(self, contract):
        """Pin the checkpoint harness's immutable identities before admission."""
        from rtp_llm.models_py.modules.dsv4.native_l20_transfer import (
            NativeL20ReplicaReceiver,
        )
        from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import V41L20Contract

        with self._worker_lock:
            self._check_worker()
            if (
                self._worker_state != "ready"
                or self._v4_args.pp_size != 2
                or not isinstance(contract, V41L20Contract)
                or self._native_pp_contract not in (None, contract)
            ):
                raise ValueError("Native V4.1 PP requires one initialized contract")
            if self._native_pp_contract is None:
                self._check_execution_config()
                contract.require_attention_config(self._v4_args.v41_config)
                self._native_pp_contract = contract
                if self._v4_args.pp_rank == 1:
                    self._native_pp_receiver = NativeL20ReplicaReceiver(
                        self._v4_args.v41_config,
                        self.kv_cache,
                        contract,
                        self.parallelism_config.tp_rank,
                    )

    def _prepare_native_pp_engine(self, inputs):
        with self._worker_lock:
            self._check_worker()
            error = None
            try:
                if self._native_pp_contract is None:
                    error = ValueError("Native V4.1 PP checkpoint contract is missing")
                self._agree_dp(
                    error,
                    (
                        "native-pp-engine-contract",
                        (
                            None
                            if self._native_pp_contract is None
                            else self._native_pp_contract.fingerprint.hex()
                        ),
                    ),
                    self._weight_device(),
                    native_dispatch=True,
                )
            finally:
                error = None
            return self.prepare_native_pp_prefill(inputs)

    @torch.inference_mode()
    def _execute_native_pp_engine(self, dispatch, transfer_round, tensors=None):
        from rtp_llm.models_py.modules.dsv4.native_pp_execution import l20_tables

        with self._worker_lock:
            self._check_worker()
            receiver = self._native_pp_receiver
            if receiver is not None:
                for identity, tables in l20_tables(dispatch).items():
                    if identity.request_id not in receiver._requests:
                        receiver.begin_request(identity, tables)
            return self._execute_native_pp_prefill(
                dispatch,
                self._native_pp_contract,
                transfer_round,
                receiver=receiver,
                tensors=tensors,
            )

    def _retire_native_pp_engine(self, identity):
        from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import V41CacheIdentity

        with self._worker_lock:
            self._check_worker()
            if self._native_pp_receiver is not None:
                self._native_pp_receiver.release(V41CacheIdentity(*identity))

    @torch.inference_mode()
    def _execute_native_pp_prefill(
        self, dispatch, contract, transfer_round, *, receiver=None, tensors=None
    ):
        """Run after native WORLD begin; the caller drains and agrees completion.

        This private adapter composes qualified owners. Public PP forwarding
        still requires executor admission, retained slots and scheduler wiring.
        """
        from rtp_llm.models_py.modules.dsv4.native_pp_execution import (
            execute_native_pp_prefill,
        )

        with self._worker_lock:
            self._check_worker()
            if self._worker_state != "ready":
                raise RuntimeError("V4.1 worker is not initialized")
            return execute_native_pp_prefill(
                self,
                dispatch,
                contract,
                transfer_round,
                receiver=receiver,
                tensors=tensors,
            )

    def _validate_decode(self, inputs, attn):
        from rtp_llm.models_py.modules.dsv4.decode.forward import (
            build_metadata_eager,
            build_paged_pool_specs,
        )

        lengths, starts = attn.input_lengths, attn.sequence_lengths
        batch = inputs.input_ids.numel()
        if (
            not (0 if self._dp_group is not None else 1)
            <= batch
            <= self._max_generate_batch_size
            or lengths is None
            or starts is None
            or lengths.shape != (batch,)
            or starts.shape != (batch,)
            or lengths.dtype not in (torch.int32, torch.int64)
            or starts.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("V4.1 decode requires one token per request")
        # Native decode input_lengths retain the original prompt lengths.
        # Token count and sequence_lengths determine the one-token batch.
        if bool(((lengths.cpu() <= 0) | (lengths.cpu() > starts.cpu())).any()):
            raise ValueError("V4.1 decode requires valid original prompt lengths")
        if bool(((starts < 1) | (starts >= min(512, self._v4_args.max_seq_len))).any()):
            raise ValueError(
                "V4.1 decode requires prompt positions within the qualified 512-token context"
            )
        positions = attn.combo_position_ids
        if (
            positions is not None
            and positions.numel()
            and (
                positions.dtype not in (torch.int32, torch.int64)
                or not torch.equal(positions.cpu(), starts.cpu())
            )
        ):
            raise ValueError("V4.1 decode positions disagree with requests")
        if batch == 0:
            return
        specs = build_paged_pool_specs(
            self.kv_cache, self.v4, self._v4_args.max_seq_len
        )
        metadata = build_metadata_eager(
            self._v4_args,
            inputs.attention_inputs,
            inputs.input_ids.device,
            specs,
            kv_cache=self.kv_cache,
            fp8_kv_cache=True,
        )
        probe = torch.empty(
            (batch, 1, self._v4_args.dim),
            device=inputs.input_ids.device,
            dtype=torch.bfloat16,
        )
        for layer in self.v4.layers:
            with bind_attn_cache(
                layer.attn, self.kv_cache, metadata.pool_block_tables, cp_ctx=None
            ):
                layer.attn._require_qualified_decode(probe, metadata)

    def _validate_forward(self, inputs, fmha_impl):
        self._check_execution_config()
        if self._v4_args.pp_size != 1:
            raise NotImplementedError(
                "V4.1 PP worker forward requires qualified native frame transport"
            )
        if fmha_impl is not None or torch.cuda.is_current_stream_capturing():
            raise NotImplementedError("V4.1 worker requires eager execution")
        if torch.cuda.current_stream(self._weight_device()) != self._worker_stream:
            raise NotImplementedError(
                "V4.1 worker requires its initialization CUDA stream"
            )
        mm = inputs.multimodal_inputs
        if (
            mm.multimodal_features
            or mm.mm_extra_input
            or (mm.mm_features_locs is not None and mm.mm_features_locs.numel())
        ):
            raise NotImplementedError("V4.1 worker image inputs are not integrated")
        if bool(
            (
                inputs.input_ids
                == self._v4_args.v41_config.get("image_token_id", 129264)
            ).any()
        ):
            raise NotImplementedError("V4.1 worker image tokens are not integrated")
        if self.kv_cache is None:
            return
        ids = inputs.input_ids
        if (
            ids.ndim != 1
            or ids.device != self._weight_device()
            or (not ids.numel() and self._dp_group is None)
            or ids.dtype not in (torch.int32, torch.int64)
            or not ids.is_contiguous()
            or bool(((ids < 0) | (ids >= self._v4_args.vocab_size)).any())
        ):
            raise ValueError("V4.1 worker requires valid token IDs (nonempty for EP1)")
        attn = primary_attention_inputs(inputs.attention_inputs, self.kv_cache)
        if (
            attn is None
            or attn.is_cuda_graph
            or getattr(attn, "is_target_verify", False)
        ):
            raise NotImplementedError("V4.1 worker graphs/verify are not integrated")
        if attn.is_prefill:
            self._validate_prefill(inputs, attn)
        else:
            self._validate_decode(inputs, attn)
        if ids.numel():
            if self._dp_group is not None and attn.is_prefill:
                tables = build_block_tables_batched(
                    self.kv_cache, inputs.attention_inputs
                )
                lengths = attn.input_lengths.to(ids.device)
                prefixes = attn.prefix_lengths.to(ids.device)
                # Use the owner's existing read-only page checks for every
                # layer before peers enter the first expert collective.
                for layer in self.v4.layers:
                    with bind_attn_cache(
                        layer.attn, self.kv_cache, tables, cp_ctx=None
                    ):
                        layer.attn._require_qualified_prefill(
                            lengths, prefixes, cp_on=False
                        )
            self._prepare_engram(inputs)
        elif (
            inputs.engram_token_windows is not None
            and inputs.engram_token_windows.numel()
        ):
            raise ValueError("V4.1 idle DP rank cannot carry token history")
        return bool(attn.is_prefill)

    @torch.inference_mode()
    def forward(self, inputs, fmha_impl: Any = None):
        with self._worker_lock:
            self._check_worker()
            if self._worker_state != "ready":
                raise RuntimeError("V4.1 worker is not initialized")
            try:
                error = None
                prefill = None
                try:
                    prefill = self._validate_forward(inputs, fmha_impl)
                except Exception as exc:
                    error = exc
                with torch.cuda.stream(self._worker_stream):
                    native_dispatch = inputs.v41_cache_spans is not None
                    self._agree_dp(
                        error,
                        (
                            "forward",
                            self.kv_cache is not None,
                            prefill,
                            native_dispatch,
                        ),
                        self._weight_device(),
                        native_dispatch=native_dispatch,
                    )
                try:
                    return super().forward(inputs, fmha_impl)
                except BaseException:
                    if self._dp_group is not None:
                        self._worker_state = "failed"
                    raise
            finally:
                self._clear_forward_state()

    def close(self):
        with self._worker_lock:
            if os.getpid() != self._worker_pid:
                raise RuntimeError("V4.1 worker cannot close inherited CUDA resources")
            if self._worker_state == "closed":
                return
            self._worker_state = "closing"
            self._clear_forward_state()
            for layer_id, table in tuple(self._engram_tables.items()):
                # close drains CUDA and retains mappings/leases on failure.
                table.close()
                del self._engram_tables[layer_id]
            for layer, _ in self._engram_layers:
                layer.engram = None
            self._engram_layers = ()
            self._engram_hash_state = None
            self._worker_state = "closed"

    def __del__(self):
        if hasattr(self, "_worker_lock") and os.getpid() == self._worker_pid:
            try:
                self.close()
            except BaseException:
                # Individual table owners retain failed CUDA teardown safely.
                pass


__all__ = ["DeepSeekV41Model"]
