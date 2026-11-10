"""Capacity-bucket workspaces and cold compilation for native eager submission.

No CUDA Graph is created. The C++ bridge launches the same five Triton programs
and patches only runtime batch counts and grids between calls.
"""

from collections import OrderedDict

import torch
import triton
from rtp_llm.models_py.sampling import rejection_graph
from rtp_llm.models_py.sampling.rejection_metadata import _prepare_rejection_metadata
from rtp_llm.models_py.sampling.vllm_rejection import (
    _compute_local_logits_stats_kernel,
    _insert_resampled_kernel,
    _rejection_kernel,
    _resample_kernel,
)

_register_plan = None
_plans = OrderedDict()
_LIMIT_BYTES = 32 << 20


class UnsupportedLaunch(ValueError):
    pass


def _stage(
    function,
    arguments,
    *,
    grid_factor,
    grid_divisor=1,
    grid_y=1,
    dynamic=None,
    num_warps=4
):
    """Compile without launching and pack only the installed Triton 3.6 ABI."""
    bound = function.signature.bind(**arguments)
    bound.apply_defaults()
    args = tuple(bound.arguments[name] for name in function.arg_names)
    kernel = function.warmup(*args, grid=(1, 1, 1), num_warps=num_warps)
    runner = kernel.run  # Load the module and configure shared memory; no launch.
    if (
        getattr(runner, "global_scratch_size", -1) != 0
        or getattr(runner, "profile_scratch_size", -1) != 0
        or getattr(kernel.metadata, "tensordesc_meta", None)
        or getattr(runner, "launch_cooperative_grid", True)
        or getattr(runner, "launch_pdl", True)
        or getattr(kernel.metadata, "num_ctas", 1) != 1
    ):
        raise UnsupportedLaunch("unsupported launch metadata")
    names = function.arg_names

    def position(key):
        key = key[0] if isinstance(key, tuple) else key
        return names.index(key) if isinstance(key, str) else key

    constants = {position(key) for key in kernel.src.constants}
    dynamic = dynamic or {}
    codes = {"i32": 1, "u32": 2, "i64": 3, "u64": 4, "fp32": 5, "fp64": 6}
    types, values, factors = [], [], []
    for key, dtype in sorted(
        kernel.src.signature.items(), key=lambda item: position(item[0])
    ):
        index = position(key)
        name = names[index]
        if index in constants or dtype == "constexpr":
            if name in dynamic:
                raise UnsupportedLaunch("runtime batch was specialized")
            continue
        if not isinstance(dtype, str):
            raise UnsupportedLaunch("non-scalar launch argument")
        code = 0 if dtype.startswith("*") else codes.get(dtype)
        if code is None:
            raise UnsupportedLaunch("unsupported launch argument type")
        value = args[index]
        if code == 0:
            value = (
                value.data_ptr() if isinstance(value, torch.Tensor) else (value or 0)
            )
        if name in dynamic and (
            code not in (1, 2, 3, 4) or name not in function.do_not_specialize
        ):
            raise UnsupportedLaunch("batch argument lacks non-specialization contract")
        types.append(code)
        values.append(value)
        factors.append(dynamic.get(name, 0))
    if len(types) > 64:
        raise UnsupportedLaunch("too many launch arguments")
    return kernel, (
        kernel.function,
        kernel.metadata.shared,
        kernel.metadata.num_warps,
        grid_factor,
        grid_divisor,
        grid_y,
        types,
        values,
        factors,
    )


class _Plan:
    def __init__(self, key):
        device, stream, forced, capacity, rows, vocab, has_cap = key
        self.key = key
        self.buffers = []

        def empty(shape, dtype=torch.float32):
            tensor = torch.empty(
                shape, device=torch.device("cuda", device), dtype=dtype
            )
            self.buffers.append(tensor)
            return tensor

        n = capacity * rows
        blocks = triton.cdiv(vocab, 8192)
        resample_blocks = triton.cdiv(vocab, 1024)
        self.addresses = empty((8 + 2 * capacity,), torch.int64)[:8]
        dummy_float = empty((1,))
        dummy_int = empty((1,), torch.int32)
        dummy_long = empty((1,), torch.int64)
        dummy_bool = empty((1,), torch.bool)
        packed, positions = empty((n,), torch.int64), empty((n,), torch.int64)
        mapping = empty((capacity,), torch.int32)
        expanded, local = empty((n,), torch.int32), empty((n,), torch.int32)
        cumulative = empty((capacity + 1,), torch.int32)
        temperature, rates = empty((capacity,)), empty((rows - 1,))
        seeds = empty((capacity,), torch.int64)
        argmax = empty((n, blocks), torch.int64)
        maxima, sums = empty((n, blocks)), empty((n, blocks))
        target_lse, draft_lse = empty((capacity,)), empty((capacity,))
        resample_ids = empty((capacity, resample_blocks), torch.int64)
        resample_max = empty((capacity, resample_blocks))
        self.bytes = sum(t.numel() * t.element_size() for t in self.buffers)
        if self.bytes > _LIMIT_BYTES:
            raise UnsupportedLaunch("workspace exceeds native cache budget")
        common = dict(
            target_logits_ptr=dummy_float,
            target_logits_stride=vocab,
            draft_logits_ptr=None,
            draft_logits_stride_0=0,
            draft_logits_stride_1=0,
            draft_logits_row_mapping_ptr=None,
            target_local_argmax_ptr=argmax,
            target_local_argmax_stride=blocks,
            target_local_max_ptr=maxima,
            target_local_max_stride=blocks,
            target_local_sumexp_ptr=sums,
            target_local_sumexp_stride=blocks,
            draft_local_max_ptr=dummy_float,
            draft_local_max_stride=blocks,
            draft_local_sumexp_ptr=dummy_float,
            draft_local_sumexp_stride=blocks,
            expanded_idx_mapping_ptr=expanded,
            expanded_local_pos_ptr=local,
            temp_ptr=temperature,
            vocab_size=vocab,
            num_logits=n,
            HAS_DRAFT_LOGITS=False,
            DRAFT_VALUES_ARE_PROBS=False,
            graph_input_addresses_ptr=self.addresses,
            sampled_ptr=dummy_int,
            sampled_stride=rows,
            rejected_steps_ptr=dummy_int,
            target_rejected_logsumexp_ptr=target_lse,
            draft_rejected_logsumexp_ptr=draft_lse,
            draft_sampled_ptr=packed,
            cu_num_logits_ptr=cumulative,
            idx_mapping_ptr=mapping,
            seed_ptr=seeds,
            pos_ptr=positions,
            synthetic_conditional_rates_ptr=rates if forced else None,
            cumulative_log_p_ptr=None,
            local_residual_mass_ptr=None,
            local_residual_mass_stride=0,
            fly_entropy_ptr=None,
            fly_entropy_threshold=0.3,
            vocab_num_blocks=blocks,
            PADDED_VOCAB_NUM_BLOCKS=triton.next_power_of_2(blocks),
            SYNTHETIC_MODE=forced,
            USE_BLOCK_VERIFICATION=False,
            FLY_WINDOW_SIZE=0,
            resampled_local_argmax_ptr=resample_ids,
            resampled_local_argmax_stride=resample_blocks,
            resampled_local_max_ptr=resample_max,
            resampled_local_max_stride=resample_blocks,
            rejected_step_ptr=dummy_int,
            contexts_ptr=None,
            contexts_stride=0,
            watermarking_ptr=None,
            watermarking_skip_mask_ptr=None,
            watermark_key_0=0,
            watermark_key_1=0,
            USE_FP64=False,
            CONTEXT_WIDTH=1,
            WATERMARK=False,
            DEDUPLICATE_CONTEXTS=False,
            SKIP_MASKED_BLOCKS=True,
            num_sampled_ptr=dummy_int,
            resample_num_blocks=resample_blocks,
            PADDED_RESAMPLE_NUM_BLOCKS=triton.next_power_of_2(resample_blocks),
            CLEAN_OUTPUT_ROWS=rows,
        )
        metadata = dict(
            draft_ids=dummy_int,
            positions=dummy_long,
            do_sample=dummy_bool,
            cap=dummy_int if has_cap else None,
            packed=packed,
            absolute_positions=positions,
            mapping=mapping,
            expanded_mapping=expanded,
            local_positions=local,
            cumulative=cumulative,
            temperature=temperature,
            rates=rates,
            count=capacity,
            rows=rows,
            draft_stride=rows - 1,
            draft_column_stride=1,
            position_stride=1,
            sample_stride=1,
            cap_stride=1 if has_cap else 0,
            BLOCK=256,
            graph_input_addresses=self.addresses,
            seed_output=seeds,
        )
        stages = [
            _stage(
                _prepare_rejection_metadata,
                metadata,
                grid_factor=rows,
                grid_divisor=256,
                dynamic={"count": 1},
            )
        ]
        for function, factor, y, block_size, dynamic, warps in (
            (
                _compute_local_logits_stats_kernel,
                rows,
                blocks,
                8192,
                {"num_logits": rows},
                4,
            ),
            (_rejection_kernel, 1, 1, None, None, 1),
            (_resample_kernel, 1, resample_blocks, 1024, None, 4),
            (_insert_resampled_kernel, 1, 1, None, None, 4),
        ):
            arguments = {
                name: common[name] for name in function.arg_names if name in common
            }
            if block_size is not None:
                arguments["BLOCK_SIZE"] = block_size
            stages.append(
                _stage(
                    function,
                    arguments,
                    grid_factor=factor,
                    grid_y=y,
                    dynamic=dynamic,
                    num_warps=warps,
                )
            )
        self.kernels = [kernel for kernel, _ in stages]
        self.ready = torch.cuda.Event()
        self.ready.record(torch.cuda.current_stream(device))
        self.capsule = _register_plan(
            self,
            self.ready.cuda_event,
            self.addresses,
            key,
            [descriptor for _, descriptor in stages],
        )


def build_plan(key):
    if (
        _register_plan is None
        or triton.__version__.split(".")[:2] != ["3", "6"]
        or torch.version.hip is not None
        or rejection_graph._try_native_guard is None
        or not rejection_graph._try_native_guard()
    ):
        return False
    try:
        if key in _plans:
            _plans.move_to_end(key)
            return True
        same_device = [old for old in _plans if old[0] == key[0]]
        while len(same_device) >= 2:
            old = same_device.pop(0)
            if not _plans[old].ready.query():
                return False
            del _plans[old]
        try:
            entry = _Plan(key)
        except UnsupportedLaunch:
            return False
        used = sum(plan.bytes for old, plan in _plans.items() if old[0] == key[0])
        if used + entry.bytes > _LIMIT_BYTES:
            return False
        _plans[key] = entry
        return True
    finally:
        rejection_graph._release_native_guard()
