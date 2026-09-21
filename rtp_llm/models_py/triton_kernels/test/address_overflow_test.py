"""Run production Triton kernels at large offsets without allocating large tensors.

The CPU interpreter executes the original device functions. Only memory and the
program id are virtualized: every active load/store must land in an explicitly
registered window. An int32 wrap therefore fails instead of accessing host RAM.
Run this test in its own process with TRITON_INTERPRET=1 (also set by BUILD).
"""

import importlib.util
import os
import unittest
from contextlib import ExitStack
from itertools import product
from pathlib import Path
from unittest.mock import patch

os.environ["TRITON_INTERPRET"] = "1"

import numpy as np
import triton.language as tl
from triton.runtime.interpreter import (
    InterpretedFunction,
    TensorHandle,
    _get_np_dtype,
    interpreter_builder,
)

from rtp_llm.models_py.triton_kernels.common.activation import _silu_and_mul_kernel
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import (
    _layer_norm_fwd_1pass_kernel,
)
from rtp_llm.models_py.triton_kernels.fla.block import (
    load_initial_state_from_block_map_kernel,
    store_ssm_state_to_block_map_kernel,
)
from rtp_llm.models_py.triton_kernels.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
)
from rtp_llm.models_py.triton_kernels.kimi_kda.chunk_intra import (
    chunk_kda_fwd_kernel_inter_solve_fused,
    chunk_kda_fwd_kernel_intra_sub_chunk,
)
from rtp_llm.models_py.triton_kernels.kimi_kda.chunk_intra_token_parallel import (
    chunk_kda_fwd_kernel_intra_token_parallel,
)
from rtp_llm.models_py.triton_kernels.kimi_kda.gate import (
    kda_gate_chunk_cumsum_vector_kernel,
)
from rtp_llm.models_py.triton_kernels.kimi_kda.wy_fast import (
    recompute_w_u_fwd_kda_kernel,
)


def device_function(kernel):
    # Bypass launch heuristics/autotuning, keeping the actual device function.
    while not isinstance(kernel, InterpretedFunction):
        kernel = kernel.fn
    return kernel


class SparseMemory:
    def __init__(self):
        self.buffers = []

    def buffer(self, name, dtype=tl.float32):
        base = (len(self.buffers) + 1) * (1 << 44)
        item = dict(name=name, base=base, windows=[])
        self.buffers.append(item)
        ty = tl.pointer_type(dtype)
        item["pointer"] = tl.tensor(TensorHandle(np.array([base], np.uint64), ty), ty)
        return item

    def window(self, buffer, offset, values):
        dtype = buffer["pointer"].dtype.element_ty
        values = np.asarray(values, dtype=_get_np_dtype(dtype)).reshape(-1).copy()
        window = dict(offset=offset, values=values, read=False, written=False)
        buffer["windows"].append(window)
        return values

    def access(self, ptrs, mask, values=None):
        dtype = ptrs.get_element_ty()
        itemsize = _get_np_dtype(dtype).itemsize
        addresses = ptrs.data.astype(np.int64)
        active = np.broadcast_to(mask.data, addresses.shape)
        result = np.zeros(addresses.shape, dtype=_get_np_dtype(dtype))
        remaining = active.copy()
        for buffer in self.buffers:
            for window in buffer["windows"]:
                start = buffer["base"] + window["offset"] * itemsize
                indices = (addresses - start) // itemsize
                selected = (
                    remaining
                    & (addresses >= start)
                    & (indices < window["values"].size)
                    & ((addresses - start) % itemsize == 0)
                )
                if np.any(selected):
                    if values is None:
                        result[selected] = window["values"][indices[selected]]
                        window["read"] = True
                    else:
                        window["values"][indices[selected]] = np.broadcast_to(
                            values, addresses.shape
                        )[selected]
                        window["written"] = True
                    remaining &= ~selected
        if np.any(remaining):
            address = int(addresses[remaining].flat[0])
            closest = min(self.buffers, key=lambda b: abs(address - b["base"]))
            offset = (address - closest["base"]) // itemsize
            raise AssertionError(
                f"Unexpected {'load' if values is None else 'store'}: "
                f"{closest['name']} element offset {offset}"
            )
        return result

    def load(self, ptrs, mask, other, *_args):
        result = self.access(ptrs, mask)
        if other is not None:
            result = np.where(mask.data, result, other.data)
        return TensorHandle(result, ptrs.get_element_ty())

    def store(self, ptrs, values, mask, *_args):
        self.access(ptrs, mask, values.data)

    def run(self, kernel, program, **kwargs):
        def program_id(axis):
            return TensorHandle(np.array([program[axis]], np.int32), tl.int32)

        with ExitStack() as stack:
            # The interpreter patches tl members, but not imported aliases.
            function = device_function(kernel)
            aliases = {"exp": lambda x: tl.exp(x), "exp2": lambda x: tl.exp2(x)}
            for name, alias in aliases.items():
                namespace = function.fn.__globals__
                if name in namespace:
                    stack.callback(namespace.__setitem__, name, namespace[name])
                else:
                    stack.callback(namespace.pop, name)
                namespace[name] = alias
            stack.enter_context(
                patch.object(interpreter_builder, "create_get_program_id", program_id)
            )
            stack.enter_context(
                patch.object(interpreter_builder, "create_masked_load", self.load)
            )
            stack.enter_context(
                patch.object(interpreter_builder, "create_masked_store", self.store)
            )
            kwargs = {
                key: value["pointer"] if isinstance(value, dict) else value
                for key, value in kwargs.items()
            }
            device_function(kernel)[(1,)](**kwargs)


class AddressOverflowTest(unittest.TestCase):
    def test_gated_norm_before_at_after_int32_boundary(self):
        for row in ((1 << 24) - 1, 1 << 24, (1 << 24) + 1):
            with self.subTest(row=row):
                mem = SparseMemory()
                x, y, w, z, rstd = [mem.buffer(n) for n in ("x", "y", "w", "z", "rstd")]
                values = np.linspace(-1, 1, 128, dtype=np.float32)
                mem.window(x, row * 128, values)
                output = mem.window(y, row * 128, np.full(128, np.nan))
                mem.window(w, 0, np.ones(128))
                mem.window(z, row * 128, np.zeros(128))
                mem.window(rstd, row, [np.nan])
                mem.run(
                    _layer_norm_fwd_1pass_kernel,
                    (row, 0, 0),
                    X=x,
                    Y=y,
                    W=w,
                    B=w,
                    Z=z,
                    Mean=rstd,
                    Rstd=rstd,
                    stride_x_row=128,
                    stride_y_row=128,
                    stride_z_row=128,
                    M=row + 1,
                    N=128,
                    eps=1e-5,
                    BLOCK_N=128,
                    HAS_BIAS=False,
                    HAS_Z=True,
                    NORM_BEFORE_GATE=True,
                    IS_RMS_NORM=True,
                    SIGMOID_GATE=True,
                )
                expected = values / np.sqrt(np.mean(values**2) + 1e-5) * 0.5
                np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-6)

    def test_moe_silu_expanded_rows(self):
        for row in ((1 << 19) - 1, 1 << 19, (1 << 20) + 1):
            with self.subTest(row=row):
                mem = SparseMemory()
                x, y = mem.buffer("gate_up"), mem.buffer("output")
                mem.window(x, row * 4096, np.full(128, 2.0))
                mem.window(x, row * 4096 + 2048, np.ones(128))
                output = mem.window(y, row * 2048, np.full(128, np.nan))
                mem.run(
                    _silu_and_mul_kernel,
                    (row, 0, 0),
                    output_ptr=y,
                    input_ptr=x,
                    N=2048,
                    input_row_stride=4096,
                    output_row_stride=2048,
                    BLOCK_SIZE_N=128,
                )
                np.testing.assert_allclose(output, 2 / (1 + np.exp(-1)), rtol=1e-6)

    def test_state_checkpoint_source_boundary(self):
        for source_index in (8191, 8192, 8193):
            with self.subTest(source_index=source_index):
                mem = SparseMemory()
                ids, prefix, cu, table = [
                    mem.buffer(n, tl.int32) for n in ("chunks", "prefix", "cu", "table")
                ]
                h, final, cache = [mem.buffer(n) for n in ("h", "final", "cache")]
                chunk_id = source_index - 1
                mem.window(ids, chunk_id * 2, [1, 3])
                mem.window(prefix, 1, [128])
                bos = (chunk_id - 3) * 64
                mem.window(cu, 1, [bos, bos + 1024])
                mem.window(table, 16 + 1, [7])
                values = np.arange(8192, dtype=np.float32) / 8192
                mem.window(h, source_index * 16 * 128 * 128, values)
                output = mem.window(cache, 7 * 262208, np.full(8192, np.nan))
                mem.run(
                    store_ssm_state_to_block_map_kernel,
                    (chunk_id, 0, 0),
                    chunk_indices=ids,
                    h=h,
                    final_states=final,
                    prefix_lengths=prefix,
                    cu_seqlens=cu,
                    block_map=table,
                    ssm_states=cache,
                    max_block_size=16,
                    HEAD_NUM=16,
                    V=128,
                    K=128,
                    BLOCK_V=64,
                    SEQ_SIZE_PER_BLOCK=256,
                    CHUNK_SIZE=64,
                    CONV_STRIDE_TOKEN=262208,
                )
                np.testing.assert_array_equal(output, values)

    def test_initial_and_final_state_batch_boundary(self):
        for batch in (8191, 8192, 8193):
            with self.subTest(batch=batch):
                mem = SparseMemory()
                prefix, table, ids, cu = [
                    mem.buffer(n, tl.int32) for n in ("prefix", "table", "chunks", "cu")
                ]
                cache, initial, h = [mem.buffer(n) for n in ("cache", "initial", "h")]
                mem.window(prefix, batch, [256])
                mem.window(table, batch * 16, [7, 9])
                values = np.arange(8192, dtype=np.float32) * 0.001
                mem.window(cache, 7 * 262208, values)
                initial_values = mem.window(
                    initial, batch * 262144, np.full(8192, np.nan)
                )
                common = dict(
                    prefix_lengths=prefix,
                    block_map=table,
                    max_block_size=16,
                    HEAD_NUM=16,
                    V=128,
                    K=128,
                    BLOCK_V=64,
                    SEQ_SIZE_PER_BLOCK=256,
                    CONV_STRIDE_TOKEN=262208,
                )
                mem.run(
                    load_initial_state_from_block_map_kernel,
                    (batch, 0, 0),
                    conv_states=cache,
                    initial_states=initial,
                    **common,
                )
                np.testing.assert_array_equal(initial_values, values)
                mem.window(ids, 0, [batch, 0])
                mem.window(cu, batch, [0, 64])
                output = mem.window(cache, 9 * 262208, np.full(8192, np.nan))
                mem.run(
                    store_ssm_state_to_block_map_kernel,
                    (0, 0, 0),
                    chunk_indices=ids,
                    h=h,
                    final_states=initial,
                    cu_seqlens=cu,
                    ssm_states=cache,
                    CHUNK_SIZE=64,
                    **common,
                )
                np.testing.assert_array_equal(output, values)

    def test_kda_ragged_and_local_token_offsets(self):
        heads, dim, chunk = 16, 128, 64
        boundary = (1 << 31) // (heads * dim)
        for start in (boundary - chunk, boundary, boundary + chunk):
            for ragged in (False, True):
                with self.subTest(start=start, ragged=ragged):
                    mem = SparseMemory()
                    cu, indices = [mem.buffer(n, tl.int32) for n in ("cu", "indices")]
                    if ragged:
                        mem.window(cu, 0, [0, start, start + chunk])
                        mem.window(indices, 0, [1, 0])
                    else:
                        mem.window(cu, 0, [0, start + chunk])
                        mem.window(indices, 0, [0, start // chunk])
                    buffers = {}
                    for name, width in (
                        ("q", dim),
                        ("k", dim),
                        ("g", dim),
                        ("v", dim),
                        ("beta", 1),
                        ("Aqk", chunk),
                        ("Akk", chunk),
                        ("Akkd", 16),
                        ("w", dim),
                        ("u", dim),
                        ("qg", dim),
                        ("kg", dim),
                        ("go", dim),
                    ):
                        buffers[name] = mem.buffer(name)
                        mem.window(
                            buffers[name],
                            start * heads * width,
                            np.zeros(chunk * heads * width),
                        )
                    alog, bias = mem.buffer("alog"), mem.buffer("bias")
                    mem.window(alog, 0, np.zeros(heads))
                    mem.window(bias, 0, np.zeros(heads * dim))
                    common = dict(
                        cu_seqlens=cu,
                        chunk_indices=indices,
                        T=start + chunk,
                        H=heads,
                        BT=chunk,
                        IS_VARLEN=True,
                    )
                    mem.run(
                        kda_gate_chunk_cumsum_vector_kernel,
                        (0, 0, 0),
                        s=buffers["g"],
                        o=buffers["go"],
                        A_log=alog,
                        dt_bias=bias,
                        scale=1.0,
                        lower_bound=-5.0,
                        S=dim,
                        BS=16,
                        REVERSE=False,
                        HAS_BIAS=True,
                        HAS_SCALE=True,
                        USE_LOWER_BOUND=True,
                        **common,
                    )
                    mem.run(
                        chunk_kda_fwd_kernel_intra_sub_chunk,
                        (0, 0, 0),
                        q=buffers["q"],
                        k=buffers["k"],
                        g=buffers["g"],
                        beta=buffers["beta"],
                        Aqk=buffers["Aqk"],
                        Akk=buffers["Akkd"],
                        scale=1.0,
                        K=dim,
                        BC=16,
                        BK=dim,
                        USE_GATHER=False,
                        **common,
                    )
                    mem.run(
                        chunk_kda_fwd_kernel_inter_solve_fused,
                        (0, 0, 0),
                        q=buffers["q"],
                        k=buffers["k"],
                        g=buffers["g"],
                        beta=buffers["beta"],
                        Aqk=buffers["Aqk"],
                        Akk=buffers["Akk"],
                        Akkd=buffers["Akkd"],
                        scale=1.0,
                        K=dim,
                        BC=16,
                        BK=32,
                        USE_SAFE_GATE=True,
                        **common,
                    )
                    mem.run(
                        recompute_w_u_fwd_kda_kernel,
                        (0, 0, 0),
                        **{
                            n: buffers[n]
                            for n in ("q", "k", "qg", "kg", "v", "beta", "w", "u")
                        },
                        A=buffers["Akk"],
                        gk=buffers["g"],
                        K=dim,
                        V=dim,
                        BK=32,
                        BV=32,
                        STORE_QG=True,
                        STORE_KG=True,
                        **common,
                    )
                    for name in ("go", "Aqk", "Akk", "w", "u", "qg", "kg"):
                        self.assertTrue(buffers[name]["windows"][0]["written"], name)

    def test_token_parallel_chunk_sizes(self):
        for chunk in (64, 128, 256):
            for ragged, start in product((False, True), (1 << 20, 1 << 27)):
                with self.subTest(chunk=chunk, ragged=ragged, start=start):
                    heads, dim = 16, 128
                    mem = SparseMemory()
                    cu = mem.buffer("cu", tl.int32)
                    mem.window(
                        cu,
                        0,
                        [0, start, start + chunk] if ragged else [0, start + chunk],
                    )
                    buffers = {}
                    for name, width in (
                        ("q", dim),
                        ("k", dim),
                        ("g", dim),
                        ("beta", 1),
                        ("Aqk", chunk),
                        ("Akk", chunk),
                    ):
                        buffers[name] = mem.buffer(name)
                        mem.window(
                            buffers[name],
                            start * heads * width,
                            np.zeros(heads * width),
                        )
                    mem.run(
                        chunk_kda_fwd_kernel_intra_token_parallel,
                        (start, 0, 0),
                        **buffers,
                        cu_seqlens=cu,
                        N=2 if ragged else 1,
                        T=start + chunk,
                        H=heads,
                        K=dim,
                        BT=chunk,
                        BC=chunk,
                        BH=1,
                        IS_VARLEN=True,
                        scale=1.0,
                    )
                    self.assertTrue(buffers["Aqk"]["windows"][0]["written"])

    def test_shared_fla_cast_before_state_multiply(self):
        mem = SparseMemory()
        heads, dim, chunk, state_index = 16, 128, 64, 8192
        cu, offsets = mem.buffer("cu", tl.int32), mem.buffer("chunk_offsets", tl.int32)
        mem.window(cu, 1, [1 << 20, (1 << 20) + chunk])
        mem.window(offsets, 1, [state_index])
        buffers = {}
        for name in ("k", "v", "w", "v_new", "gk"):
            buffers[name] = mem.buffer(name)
            mem.window(
                buffers[name], (1 << 20) * heads * dim, np.zeros(chunk * heads * dim)
            )
        h, h0, ht, g = [mem.buffer(n) for n in ("h", "h0", "ht", "g")]
        mem.window(h, state_index * heads * dim * dim, np.full(dim * dim, np.nan))
        mem.window(h0, heads * dim * dim, np.zeros(dim * dim))
        mem.window(ht, heads * dim * dim, np.full(dim * dim, np.nan))
        mem.run(
            chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
            (0, heads, 0),
            **buffers,
            g=g,
            h=h,
            h0=h0,
            ht=ht,
            cu_seqlens=cu,
            chunk_offsets=offsets,
            T=(1 << 20) + chunk,
            H=heads,
            Hg=heads,
            K=dim,
            V=dim,
            BT=chunk,
            BV=32,
            USE_G=False,
            USE_GK=True,
            USE_INITIAL_STATE=True,
            STORE_FINAL_STATE=True,
            SAVE_NEW_VALUE=True,
            IS_VARLEN=True,
            USE_EXP2=True,
        )
        self.assertTrue(h["windows"][0]["written"])

    def test_small_page_indexer_output_offset(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "modules/dsv4/fp8/_indexer_small_page.py"
        )
        spec = importlib.util.spec_from_file_location("small_page_address_test", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for batch in (8191, 8192, 8193):
            with self.subTest(batch=batch):
                mem = SparseMemory()
                q = mem.buffer("q", tl.uint8)
                w, out = mem.buffer("weights"), mem.buffer("scores")
                table, lengths = mem.buffer("table", tl.int32), mem.buffer(
                    "lengths", tl.int32
                )
                cache = mem.buffer("cache", tl.uint8)
                mem.window(q, batch * 16 * 128, np.zeros(16 * 128))
                mem.window(w, batch * 16, np.ones(16))
                mem.window(lengths, batch, [0])
                # All keys masked: this still must initialize the large-offset output.
                output = mem.window(out, batch * (1 << 18), np.full(128, np.nan))
                mem.run(
                    module._score,
                    (batch, 0, 0),
                    Q=q,
                    W=w,
                    Cache=cache,
                    Table=table,
                    Lengths=lengths,
                    Out=out,
                    NEXT=1,
                    HEADS=16,
                    HEAD_TILE=16,
                    PAGE=16,
                    TABLE_STRIDE=16384,
                    WIDTH=1 << 18,
                    QUERY_TILE=1,
                    KEY_TILE=128,
                )
                np.testing.assert_array_equal(output, np.full(128, -np.inf))


if __name__ == "__main__":
    unittest.main()
