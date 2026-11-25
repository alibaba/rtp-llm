"""Module-owned V4 FP8-activation/FP4-expert checkpoints.

This is the DeepGEMM group-32 UE8M0 recipe, not ModelOpt/NVFP4 group-16.
MoE weights are EP-sharded and attention-TP-replicated, matching current main.
"""

from typing import Optional

import torch
from torch import nn

from rtp_llm.models_py.module_base import RtpModule, copy_weight_
from rtp_llm.models_py.read_spec import ReadSpec, unwrap_checkpoint_tensor


class OwnedFp8Fp4MoE(RtpModule):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        moe_inter_dim: int,
        n_routed_experts: int,
        n_activated_experts: int,
        n_shared_experts: int,
        score_func: str = "sqrtsoftplus",
        route_scale: float = 1.0,
        swiglu_limit: float = 0.0,
        n_hash_layers: int = 0,
        vocab_size: int = 0,
        *,
        prefix: str = "",
        tp_size: int = 1,
        tp_rank: int = 0,
        ep_size: int = 1,
        ep_rank: int = 0,
        world_size: Optional[int] = None,
        world_rank: Optional[int] = None,
        max_tokens_per_rank: int = 8192,
        is_decode_role: bool = False,
        strategy: str = "auto",
        moe_strategy: Optional[str] = None,
        n_physical_experts: Optional[int] = None,
        chunking_enabled: bool = True,
        model_type: str = "deepseek_v4",
        shared_fp8_block_size: int = 128,
        observer_factory=None,
        record_function_scope=None,
    ):
        super().__init__()
        if not 0 <= tp_rank < tp_size or not 0 <= ep_rank < ep_size:
            raise ValueError("FP8/FP4 MoE TP/EP rank must be within positive size")
        if dim <= 0 or moe_inter_dim <= 0 or dim % 32 or moe_inter_dim % 32:
            raise ValueError("FP8/FP4 MoE dimensions must be positive multiples of 32")
        if n_routed_experts <= 0 or n_routed_experts % ep_size:
            raise ValueError(
                "FP8/FP4 expert count must be positive and divisible by EP"
            )
        if not 1 <= n_activated_experts <= n_routed_experts:
            raise ValueError("FP8/FP4 top-k must be within the logical expert count")
        if n_shared_experts < 0 or shared_fp8_block_size not in (32, 128):
            raise ValueError("Invalid FP8 shared-expert count or block size")
        if n_physical_experts not in (None, n_routed_experts):
            raise ValueError(
                "Current main FP8_FP4 strategies do not support redundant experts"
            )
        self.prefix = prefix.rstrip(".")
        self.layer_id = layer_id
        self.dim = dim
        self.moe_inter_dim = moe_inter_dim
        self.num_experts = n_routed_experts
        self.n_local_experts = n_routed_experts // ep_size
        self.ep_size, self.ep_rank = ep_size, ep_rank
        self.tp_size, self.tp_rank = tp_size, tp_rank
        self.expert_start = self.n_local_experts * ep_rank
        self.shared_fp8_block_size = shared_fp8_block_size
        self._shared_inter = moe_inter_dim * n_shared_experts
        self._hash_gate = layer_id < n_hash_layers
        self._loaded = set()
        self._runtime = None
        self._eplb_stats = None
        self._runtime_configured = False
        self._runtime_transfer_started = False
        self._runtime_kwargs = dict(
            layer_id=layer_id,
            dim=dim,
            moe_inter_dim=moe_inter_dim,
            n_routed_experts=n_routed_experts,
            n_activated_experts=n_activated_experts,
            n_shared_experts=n_shared_experts,
            score_func=score_func,
            route_scale=route_scale,
            swiglu_limit=swiglu_limit,
            n_hash_layers=n_hash_layers,
            vocab_size=vocab_size,
            ep_size=ep_size,
            ep_rank=ep_rank,
            world_size=world_size,
            world_rank=world_rank,
            max_tokens_per_rank=max_tokens_per_rank,
            is_decode_role=is_decode_role,
            strategy=moe_strategy or strategy,
            n_physical_experts=n_physical_experts,
            chunking_enabled=chunking_enabled,
            model_type=model_type,
            moe_w1_layout="gate_up",
            observer_factory=observer_factory,
        )
        if record_function_scope is not None:
            self._runtime_kwargs["record_function_scope"] = record_function_scope
        e, m, h = self.n_local_experts, moe_inter_dim, dim
        self._parameter("routed_w13", (e, 2 * m, h // 2), torch.int8)
        self._parameter("routed_w2", (e, h, m // 2), torch.int8)
        self._parameter("routed_s13", (e, 2 * m, h // 32), torch.float8_e8m0fnu)
        self._parameter("routed_s2", (e, h, m // 32), torch.float8_e8m0fnu)
        self._parameter("router_weight", (n_routed_experts, h), torch.bfloat16)
        if self._hash_gate:
            if vocab_size <= 0:
                raise ValueError("Hash FP8/FP4 MoE requires a positive vocabulary size")
            self._parameter(
                "router_tid2eid", (vocab_size, n_activated_experts), torch.int32
            )
        else:
            self._parameter("router_bias", (n_routed_experts,), torch.float32)
        if n_shared_experts:
            s, b = self._shared_inter, shared_fp8_block_size
            self._parameter("shared_w13", (2 * s, h), torch.float8_e4m3fn)
            self._parameter("shared_w2", (h, s), torch.float8_e4m3fn)
            self._parameter(
                "shared_s13",
                (2 * self._ceil(s, b), self._ceil(h, b)),
                torch.float8_e8m0fnu,
            )
            self._parameter(
                "shared_s2", (self._ceil(h, b), self._ceil(s, b)), torch.float8_e8m0fnu
            )
        self._expected = set()
        for expert in range(self.expert_start, self.expert_start + e):
            for projection in ("w1", "w2", "w3"):
                for suffix in ("weight", "scale"):
                    self._expected.add(f"experts.{expert}.{projection}.{suffix}")
        if n_shared_experts:
            for projection in ("w1", "w2", "w3"):
                for suffix in ("weight", "scale"):
                    self._expected.add(f"shared_experts.{projection}.{suffix}")
        self._expected.update(
            ("gate.weight", "gate.tid2eid" if self._hash_gate else "gate.bias")
        )

    @staticmethod
    def _ceil(value, block):
        return (value + block - 1) // block

    def _parameter(self, name, shape, dtype):
        self.register_parameter(
            name, nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
        )

    def _relative_name(self, name):
        if self.prefix and name.startswith(self.prefix + "."):
            return name[len(self.prefix) + 1 :]
        return name

    def _target(self, name):
        name = self._relative_name(name)
        pieces = name.split(".")
        if pieces[0] == "gate" and len(pieces) == 2:
            key = {
                "weight": "router_weight",
                "bias": "router_bias",
                "tid2eid": "router_tid2eid",
            }.get(pieces[1])
            if key is not None and hasattr(self, key):
                return getattr(self, key)
        if pieces[0] == "experts" and len(pieces) == 4:
            expert, projection, suffix = int(pieces[1]), pieces[2], pieces[3]
            if not 0 <= expert < self.num_experts:
                raise ValueError(f"Expert outside checkpoint range: {name}")
            if projection not in ("w1", "w2", "w3") or suffix not in (
                "weight",
                "scale",
            ):
                raise ValueError(f"Unknown FP8/FP4 checkpoint tensor: {name}")
            if (
                not self.expert_start
                <= expert
                < self.expert_start + self.n_local_experts
            ):
                return None
            local = expert - self.expert_start
            target = getattr(
                self,
                "routed_"
                + ("s" if suffix == "scale" else "w")
                + ("2" if projection == "w2" else "13"),
            )[local]
            if projection != "w2":
                target = target.narrow(
                    0,
                    self.moe_inter_dim if projection == "w3" else 0,
                    self.moe_inter_dim,
                )
            return target
        if pieces[0] == "shared_experts" and len(pieces) == 3 and self._shared_inter:
            projection, suffix = pieces[1:]
            if projection not in ("w1", "w2", "w3") or suffix not in (
                "weight",
                "scale",
            ):
                raise ValueError(f"Unknown shared FP8 checkpoint tensor: {name}")
            target = getattr(
                self,
                "shared_"
                + ("s" if suffix == "scale" else "w")
                + ("2" if projection == "w2" else "13"),
            )
            if projection != "w2":
                rows = (
                    self._ceil(self._shared_inter, self.shared_fp8_block_size)
                    if suffix == "scale"
                    else self._shared_inter
                )
                target = target.narrow(0, rows if projection == "w3" else 0, rows)
            return target
        raise ValueError(f"Unknown FP8/FP4 checkpoint tensor: {name}")

    def checkpoint_read_spec(self, name, shape):
        target = self._target(name)
        if target is None:
            return ReadSpec.skip(name, tuple(shape))
        if tuple(target.shape) != tuple(shape):
            raise ValueError(
                f"Shape mismatch for {name}: expected {tuple(target.shape)}, got {shape}"
            )
        return ReadSpec.full(name, tuple(shape))

    def load_weights(self, weights):
        for name, value in weights.items():
            name = self._relative_name(name)
            target = self._target(name)
            if target is None:
                continue
            if name in self._loaded:
                raise RuntimeError(f"Duplicate FP8/FP4 checkpoint tensor: {name}")
            tensor, read_spec = unwrap_checkpoint_tensor(value)
            if read_spec is not None and read_spec.is_local_shard:
                raise ValueError(
                    "FP8/FP4 per-expert checkpoint leaves must remain whole"
                )
            if name == "gate.tid2eid":
                # V4 release tables use I64; the gate ABI consumes I32. Only
                # expert identifiers, never FP4/FP8 data, can be narrowed.
                if tensor.dtype not in (torch.int64, torch.int32):
                    raise TypeError(f"Invalid {name} dtype: {tensor.dtype}")
                if tensor.shape != target.shape:
                    raise ValueError(f"Shape mismatch for {name}: {tensor.shape}")
                lo, hi = torch.aminmax(tensor)
                lo, hi = lo.item(), hi.item()
                if lo < 0 or hi >= self.num_experts:
                    raise ValueError(
                        f"gate.tid2eid contains an out-of-range expert index: layer={self.layer_id} "
                        f"dtype={tensor.dtype} shape={tuple(tensor.shape)} "
                        f"observed=[{lo},{hi}] valid=[0,{self.num_experts})"
                    )
                with torch.no_grad():
                    target.copy_(tensor)
            else:
                copy_weight_(target, tensor, name)
            self._loaded.add(name)

    def finish_resident_checkpoint_restore(self):
        # All packed rank-local destination tensors were just shape/type checked.
        # Only a fully validated cold load can publish these fused expert stacks.
        self._loaded = self._expected.copy()

    def checkpoint_load_complete(self):
        return self._expected.issubset(self._loaded)

    def validate_weights_loaded(self, loaded_tensor_ids=None):
        missing = self._expected - self._loaded
        if missing:
            raise RuntimeError(
                f"Missing FP8/FP4 MoE checkpoint tensors: {sorted(missing)[:8]}"
            )

    def validate_runtime_device(self, device):
        device = torch.device(device)
        if device.type == "cpu":
            return  # Native checkpoint staging, never an execution fallback.
        if device.type != "cuda" or torch.version.hip is not None:
            raise RuntimeError("FP8_FP4 MoE execution requires NVIDIA CUDA")
        major = torch.cuda.get_device_capability(device)[0]
        if (self.ep_size == 1 and major != 10) or (self.ep_size > 1 and major < 10):
            raise RuntimeError(
                "Current main FP8_FP4 requires SM100 (EP: SM100 or newer)"
            )
        import importlib

        try:
            backend = importlib.import_module("deep_gemm")
        except Exception as error:
            raise RuntimeError("FP8_FP4 MoE cannot import DeepGEMM") from error
        required = ["transform_sf_into_required_layout"]
        strategy = self._runtime_kwargs["strategy"]
        if self.ep_size > 1:
            required += [
                "fp8_fp4_mega_moe",
                "get_symm_buffer_for_mega_moe",
                "transform_weights_for_mega_moe",
            ]
        elif strategy == "grouped_fp4":
            required += [
                "fp8_fp4_gemm_nt",
                "m_grouped_fp8_fp4_gemm_nt_contiguous",
                "get_mk_alignment_for_contiguous_layout",
            ]
        elif strategy == "local_loop":
            required += ["fp8_fp4_gemm_nt"]
        else:
            # Both local-loop and grouped's small-batch fallback consume this.
            required += ["fp8_fp4_gemm_nt"]
        missing = [
            name for name in required if not callable(getattr(backend, name, None))
        ]
        if missing:
            raise RuntimeError(
                f"DeepGEMM FP8_FP4 is missing required symbols: {missing}"
            )

    def process_weights_after_loading(self):
        self.validate_weights_loaded()
        # Role and the token budget are established later by model.initialize.
        # Never allocate dispatch/symmetric-memory buffers while loading weights.

    @property
    def layer_idx(self):
        return self.layer_id

    def bind_eplb_stats(self, log_stats, gpu_loads, live_rows=None):
        from rtp_llm.models_py.layers.eplb_stats import ExpertRouteStats

        self._eplb_stats = ExpertRouteStats(
            log_stats, gpu_loads, num_experts=self.num_experts,
            ep_size=self.ep_size, device=self.router_weight.device, live_rows=live_rows,
        )
        if self._runtime is not None:
            self._runtime.set_route_stats(self._eplb_stats)

    def configure_runtime(self, *, is_decode_role, max_tokens_per_rank, **kwargs):
        allowed = {
            "world_size",
            "world_rank",
            "chunking_enabled",
            "observer_factory",
            "record_function_scope",
        }
        unknown = set(kwargs) - allowed
        if unknown:
            raise ValueError(
                f"Unknown FP8/FP4 runtime configuration: {sorted(unknown)}"
            )
        if not isinstance(is_decode_role, bool) or max_tokens_per_rank <= 0:
            raise ValueError(
                "FP8/FP4 runtime requires a bool role and positive token budget"
            )
        values = dict(
            kwargs,
            is_decode_role=is_decode_role,
            max_tokens_per_rank=int(max_tokens_per_rank),
        )
        if self._runtime is not None and any(
            self._runtime_kwargs.get(k) != v for k, v in values.items()
        ):
            raise RuntimeError("Cannot reconfigure a live FP8/FP4 executor")
        self._runtime_kwargs.update(values)
        self._runtime_configured = True

    def initialize_runtime(self):
        if self._runtime is not None:
            return
        if self._runtime_transfer_started:
            raise RuntimeError(
                "FP8/FP4 runtime initialization failed after weight transfer; "
                "reload the model"
            )
        if not self._runtime_configured:
            raise RuntimeError(
                "FP8/FP4 MoE requires configure_runtime after model role initialization"
            )
        self.validate_weights_loaded()
        if self.routed_w13.device.type == "cpu":
            raise RuntimeError(
                "FP8/FP4 MoE requires accelerator post-load runtime initialization"
            )
        self.validate_runtime_device(self.routed_w13.device)
        from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer import (
            ChunkedFp8Fp4MoeLayer,
        )

        self._runtime = ChunkedFp8Fp4MoeLayer(weight_owner=self, **self._runtime_kwargs)
        if self._eplb_stats is not None:
            self._runtime.set_route_stats(self._eplb_stats)

    def take_executor_weight(self, name):
        """Transfer the original packed parameter once, without copying.

        MegaMoE serializes its transforms to bound memory. Retaining a second
        reference here would keep the old storage alive after each transform.
        """
        if not name.startswith(("routed_", "shared_")) or name not in self._parameters:
            raise KeyError(name)
        weight = self._parameters[name]
        if weight is None:
            raise RuntimeError(f"FP8/FP4 executor already took {name}")
        self._parameters[name] = None
        self._runtime_transfer_started = True
        return weight

    @property
    def fused_moe(self):
        if self._runtime is None:
            raise RuntimeError(
                "FP8/FP4 executor is unavailable before initialize_runtime"
            )
        return self._runtime.fused_moe

    @property
    def strategy_name(self):
        return (
            self._runtime_kwargs["strategy"]
            if self._runtime is None
            else self._runtime.strategy_name
        )

    @property
    def max_tokens_per_rank(self):
        return self._runtime_kwargs["max_tokens_per_rank"]

    @property
    def chunking_enabled(self):
        return self._runtime_kwargs["chunking_enabled"]

    @property
    def _is_decode_role(self):
        return self._runtime_kwargs["is_decode_role"]

    @property
    def gate(self):
        return None if self._runtime is None else self._runtime.gate

    @property
    def shared_experts(self):
        return None if self._runtime is None else self._runtime.shared_experts

    def forward(self, x, input_ids=None, **kwargs):
        if self._runtime is None:
            self.initialize_runtime()
        return self._runtime(x, input_ids, **kwargs)
