"""Rank-local V4 MXFP4 parameters for the M890P provider, never a CUDA fallback."""

import torch

from rtp_llm.models_py.new_models.deepseek_v4.moe import OwnedFp8Fp4MoE
from rtp_llm.models_py.read_spec import (
    ReadSpec,
    make_sharded_read_spec,
    unwrap_checkpoint_tensor,
)


def create_ppu_moe(args, layer_id, platform_provider):
    return OwnedPpuMoE(args, layer_id, platform_provider)


class OwnedPpuMoE(OwnedFp8Fp4MoE):
    def __init__(self, args, layer_id, platform_provider):
        if platform_provider is None:
            raise ValueError("PPU MoE requires the model's bound provider")
        if args.n_shared_experts != 1:
            raise ValueError("PPU MoE requires one shared expert")
        self._pure_tp = args.ep_size == 1 and args.tp_size > 1 and args.dp_size == 1
        self._global_inter = args.moe_inter_dim
        divisor = args.tp_size if self._pure_tp else 1
        if args.ep_size == 1 and args.dim % 512:
            raise ValueError("PPU grouped MXFP4 requires dim aligned to 512")
        if args.moe_inter_dim % (128 * divisor) or args.dim % 128:
            raise ValueError(
                "PPU shared weights require block-128 aligned rank-local geometry"
            )
        super().__init__(
            layer_id=layer_id,
            dim=args.dim,
            moe_inter_dim=args.moe_inter_dim // divisor,
            n_routed_experts=args.n_routed_experts,
            n_activated_experts=args.n_activated_experts,
            n_shared_experts=args.n_shared_experts,
            score_func=args.score_func,
            route_scale=args.route_scale,
            swiglu_limit=args.swiglu_limit,
            n_hash_layers=args.n_hash_layers,
            vocab_size=args.vocab_size,
            tp_size=args.tp_size,
            tp_rank=args.tp_rank,
            ep_size=args.ep_size,
            ep_rank=args.ep_rank,
            prefix=f"layers.{layer_id}.ffn",
            n_physical_experts=args.n_physical_experts,
        )
        self._provider = platform_provider

    def _parameter(self, name, shape, dtype):
        if self._pure_tp and name in ("shared_s13", "shared_s2"):
            # PpuTPSharedExpert consumes already-prepared float scales without a
            # duplicate converted copy. The checkpoint stores UE8M0.
            dtype = torch.float32
        super()._parameter(name, shape, dtype)

    def _split_dim(self, name):
        pieces = self._relative_name(name).split(".")
        if not self._pure_tp or pieces[0] == "gate":
            return None
        return 1 if pieces[-2] == "w2" else 0

    def checkpoint_read_spec(self, name, shape):
        shape = tuple(shape)
        target = self._target(name)
        if target is None:
            return ReadSpec.skip(name, shape)
        axis = self._split_dim(name)
        if axis is None:
            return super().checkpoint_read_spec(name, shape)
        full_shape = list(target.shape)
        full_shape[axis] *= self.tp_size
        if tuple(full_shape) != shape:
            raise ValueError(
                f"Shape mismatch for {name}: expected {tuple(full_shape)}, got {shape}"
            )
        return make_sharded_read_spec(name, shape, axis, self.tp_size, self.tp_rank)

    def load_weights(self, weights):
        for name, value in weights.items():
            name = self._relative_name(name)
            axis = self._split_dim(name)
            if axis is None:
                super().load_weights({name: value})
                continue
            target = self._target(name)
            tensor, proof = unwrap_checkpoint_tensor(value)
            if proof is None:
                spec = self.checkpoint_read_spec(name, tensor.shape)
                tensor, _ = unwrap_checkpoint_tensor(spec.materialize(tensor))
            else:
                expected = self.checkpoint_read_spec(name, proof.source_shape)
                if (
                    proof.slices != expected.slices
                    or proof.shard_count != expected.shard_count
                    or proof.shard_rank != expected.shard_rank
                    or proof.shard_dim != axis
                ):
                    raise ValueError(f"Invalid PPU shard proof for {name}")
            if tuple(tensor.shape) != tuple(target.shape):
                raise ValueError(f"Local PPU shape mismatch for {name}")
            if name.startswith("shared_experts.") and name.endswith(".scale"):
                if tensor.dtype != torch.float8_e8m0fnu:
                    raise TypeError(f"Expected raw UE8M0 shared scale for {name}")
                tensor = tensor.to(torch.float32)
            super().load_weights({name: tensor})

    def validate_runtime_device(self, device):
        if torch.device(device).type == "cpu":
            return  # model parameter staging only
        from rtp_llm.device.device_type import is_ppu

        if torch.device(device).type != "cuda" or not is_ppu():
            raise RuntimeError(
                "V4 PPU MoE requires M890P; CUDA/HIP executors are not interchangeable"
            )

    def initialize_runtime(self):
        if self._runtime is None:
            raise RuntimeError(
                "PPU MoE requires bind_runtime(args) after model role initialization"
            )

    def bind_runtime(self, args):
        if self._runtime is not None:
            return self._runtime
        if self._runtime_transfer_started:
            raise RuntimeError(
                "PPU MoE bind failed after weight transfer; reload the model"
            )
        self.validate_weights_loaded()
        if self.routed_w13.device.type == "cpu":
            raise RuntimeError("PPU MoE runtime binding requires PPU-resident weights")
        self.validate_runtime_device(self.routed_w13.device)
        from rtp_llm.models_py.modules.dsv4 import _profiler

        if (
            args.tp_size,
            args.tp_rank,
            args.ep_size,
            args.ep_rank,
            args.moe_inter_dim,
        ) != (
            self.tp_size,
            self.tp_rank,
            self.ep_size,
            self.ep_rank,
            self._global_inter,
        ):
            raise ValueError("PPU runtime topology differs from loaded MoE weights")

        def no_cuda_fallback(**kwargs):
            raise RuntimeError(
                "PPU MoE provider did not select a native implementation"
            )

        self._runtime = self._provider.build_moe(
            no_cuda_fallback,
            layer_id=self.layer_id,
            dim=args.dim,
            moe_inter_dim=args.moe_inter_dim,
            n_routed_experts=args.n_routed_experts,
            n_activated_experts=args.n_activated_experts,
            n_shared_experts=args.n_shared_experts,
            score_func=args.score_func,
            route_scale=args.route_scale,
            swiglu_limit=args.swiglu_limit,
            n_hash_layers=args.n_hash_layers,
            vocab_size=args.vocab_size,
            layer_weights=self,
            tp_size=args.tp_size,
            tp_rank=args.tp_rank,
            ep_size=args.ep_size,
            ep_rank=args.ep_rank,
            world_size=args.world_size,
            world_rank=args.world_rank,
            max_tokens_per_rank=args.max_tokens_per_rank,
            is_decode_role=args.is_decode_role,
            strategy=args.moe_strategy,
            n_physical_experts=args.n_physical_experts,
            record_function_scope=_profiler.moe_record_function_scope,
        )
        return self._runtime
