#
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""BF16 GDN projections for the measured H20 TP2 matrix shapes."""

import ctypes as ct
import threading
import weakref
from functools import lru_cache

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.models_py.utils.device_capability import cached_device_capability


@triton.jit
def _prefill_projection(
    X, W, Y, N: tl.constexpr, K: tl.constexpr, WN: tl.constexpr, WK: tl.constexpr
):
    # The grouped traversal follows Triton's MIT-licensed matmul tutorial.
    # Copyright 2018-2020 Philippe Tillet; Copyright 2020-2022 OpenAI.
    tile = tl.program_id(0)
    tiles_n = tl.cdiv(N, 128)
    group = tile // (8 * tiles_n)
    row_tile = group * 8 + tile % 8
    col_tile = (tile % (8 * tiles_n)) // 8
    rows = row_tile * 64 + tl.arange(0, 64)
    cols = col_tile * 128 + tl.arange(0, 128)
    inner = tl.arange(0, 64)
    accumulator = tl.zeros((64, 128), tl.float32)
    for start in range(tl.cdiv(K, 64)):
        kk = start * 64 + inner
        x = tl.load(X + rows[:, None] * K + kk[None, :], kk[None, :] < K, 0)
        w = tl.load(
            W + cols[None, :] * WN + kk[:, None] * WK,
            (cols[None, :] < N) & (kk[:, None] < K),
            0,
        )
        accumulator = tl.dot(x, w, accumulator)
    tl.store(Y + rows[:, None] * N + cols[None, :], accumulator, cols[None, :] < N)


class _Algorithm(ct.Structure):
    _fields_ = [("data", ct.c_uint64 * 8)]


class _Heuristic(ct.Structure):
    _fields_ = [
        ("algorithm", _Algorithm),
        ("workspace_size", ct.c_size_t),
        ("state", ct.c_int),
        ("waves", ct.c_float),
        ("reserved", ct.c_int * 4),
    ]


def _check(status):
    if status:
        raise RuntimeError(f"GDN cuBLASLt returned status {status}")


@lru_cache(maxsize=1)
def _library():
    try:
        lib = ct.CDLL("libcublasLt.so.13")
    except OSError:
        return None
    lib.cublasLtGetVersion.restype = ct.c_size_t
    # Tactics and heuristic positions were measured against cuBLASLt 13.1.0.
    if lib.cublasLtGetVersion() != 130100:
        return None
    pointer = ct.c_void_p
    signatures = {
        "cublasLtCreate": [ct.POINTER(pointer)],
        "cublasLtMatmulDescCreate": [ct.POINTER(pointer), ct.c_int, ct.c_int],
        "cublasLtMatmulDescSetAttribute": [pointer, ct.c_int, pointer, ct.c_size_t],
        "cublasLtMatrixLayoutCreate": [
            ct.POINTER(pointer),
            ct.c_int,
            ct.c_uint64,
            ct.c_uint64,
            ct.c_int64,
        ],
        "cublasLtMatrixLayoutSetAttribute": [pointer, ct.c_int, pointer, ct.c_size_t],
        "cublasLtMatmulPreferenceCreate": [ct.POINTER(pointer)],
        "cublasLtMatmulPreferenceSetAttribute": [
            pointer,
            ct.c_int,
            pointer,
            ct.c_size_t,
        ],
        "cublasLtMatmulAlgoGetHeuristic": [pointer] * 7
        + [ct.c_int, ct.POINTER(_Heuristic), ct.POINTER(ct.c_int)],
        "cublasLtMatmulAlgoCheck": [pointer] * 8,
        "cublasLtMatmulAlgoCapGetAttribute": [
            pointer,
            ct.c_int,
            pointer,
            ct.c_size_t,
            ct.POINTER(ct.c_size_t),
        ],
        "cublasLtMatmul": [pointer] * 14 + [ct.c_size_t, pointer],
    }
    for name, arguments in signatures.items():
        function = getattr(lib, name)
        function.argtypes = arguments
        function.restype = ct.c_int
    return lib


def _release(library, resources):
    for name, pointer in reversed(resources):
        getattr(library, name)(pointer)


class _MatmulPlan:
    """Immutable descriptors; every invocation owns its CUDA workspace."""

    def __init__(self, weight, rows, is_input):
        self.library = lib = _library()
        self.resources = resources = []
        self.cleanup = weakref.finalize(self, _release, lib, resources)
        self.rows, self.columns, self.inner = rows, weight.shape[0], weight.shape[1]
        self.swapped = is_input and rows == 64
        self.alpha, self.beta = ct.c_float(1), ct.c_float(0)
        self.valid = False

        def create(function, destroy, *args):
            pointer = ct.c_void_p()
            _check(getattr(lib, function)(ct.byref(pointer), *args))
            resources.append((destroy, pointer))
            return pointer

        self.handle = create("cublasLtCreate", "cublasLtDestroy")
        self.operation = create(
            "cublasLtMatmulDescCreate", "cublasLtMatmulDescDestroy", 68, 0
        )
        transpose = ct.c_int(1)
        _check(
            lib.cublasLtMatmulDescSetAttribute(
                self.operation, 4, ct.byref(transpose), 4
            )
        )

        def layout(m, n, leading, row_major):
            result = create(
                "cublasLtMatrixLayoutCreate",
                "cublasLtMatrixLayoutDestroy",
                14,
                m,
                n,
                leading,
            )
            order = ct.c_int(int(row_major))
            _check(lib.cublasLtMatrixLayoutSetAttribute(result, 1, ct.byref(order), 4))
            return result

        self.input_layout = layout(rows, self.inner, self.inner, True)
        self.weight_layout = layout(
            self.columns,
            self.inner,
            weight.stride(0) if weight.stride(1) == 1 else weight.stride(1),
            weight.stride(1) == 1,
        )
        if self.swapped:
            self.input_layout, self.weight_layout = (
                self.weight_layout,
                self.input_layout,
            )
            self.output_layout = layout(self.columns, rows, self.columns, False)
        else:
            self.output_layout = layout(rows, self.columns, self.columns, True)
        result = _Heuristic()
        if rows == 64:
            # Opaque records from the measured 13.1.0 library. The version guard
            # and AlgoCheck keep these local to the matching library and layout.
            words = (
                (
                    446676598850,
                    4294967331,
                    0,
                    844424930131968,
                    558775245209601,
                    60129542158,
                    292058693646,
                    0,
                )
                if is_input
                else (
                    532575944770,
                    8589934627,
                    2,
                    844424930131968,
                    558775245209601,
                    60129542158,
                    292058693646,
                    0,
                )
            )
            algorithm = _Algorithm((ct.c_uint64 * 8)(*words))
            status = lib.cublasLtMatmulAlgoCheck(
                self.handle,
                self.operation,
                self.input_layout,
                self.weight_layout,
                self.output_layout,
                self.output_layout,
                ct.byref(algorithm),
                ct.byref(result),
            )
            if status or result.state:
                return
            result.algorithm = algorithm
        else:
            preference = create(
                "cublasLtMatmulPreferenceCreate", "cublasLtMatmulPreferenceDestroy"
            )
            budget = ct.c_size_t(64 * 1024 * 1024)
            _check(
                lib.cublasLtMatmulPreferenceSetAttribute(
                    preference, 1, ct.byref(budget), ct.sizeof(budget)
                )
            )
            choices, count = (_Heuristic * 24)(), ct.c_int()
            _check(
                lib.cublasLtMatmulAlgoGetHeuristic(
                    self.handle,
                    self.operation,
                    self.input_layout,
                    self.weight_layout,
                    self.output_layout,
                    self.output_layout,
                    preference,
                    24,
                    choices,
                    ct.byref(count),
                )
            )
            usable = [choice for choice in choices[: count.value] if choice.state == 0]
            index = 7 if rows == 32768 else 1
            if len(usable) <= index:
                return
            result = usable[index]
        self.result = result
        alignments = []
        for attribute in range(16, 20):  # MIN_ALIGNMENT_{A,B,C,D}_BYTES
            alignment, written = ct.c_uint32(), ct.c_size_t()
            _check(
                lib.cublasLtMatmulAlgoCapGetAttribute(
                    ct.byref(result.algorithm),
                    attribute,
                    ct.byref(alignment),
                    ct.sizeof(alignment),
                    ct.byref(written),
                )
            )
            alignments.append(alignment.value)
        self.alignments = tuple(alignments)
        self.valid = all(self.alignments) and result.workspace_size <= 64 * 1024 * 1024

    def __call__(self, value, weight):
        left, right = (weight, value) if self.swapped else (value, weight)
        inputs = (left.data_ptr(), right.data_ptr())
        if any(
            address % alignment
            for address, alignment in zip(inputs, self.alignments[:2])
        ):
            return F.linear(value, weight)
        output = value.new_empty((self.rows, self.columns))
        output_ptr = output.data_ptr()
        if any(output_ptr % alignment for alignment in self.alignments[2:]):
            del output
            return F.linear(value, weight)
        pointers = (*inputs, output_ptr)
        workspace = torch.empty(
            self.result.workspace_size, dtype=torch.uint8, device=value.device
        )
        _check(
            self.library.cublasLtMatmul(
                self.handle,
                self.operation,
                ct.byref(self.alpha),
                pointers[0],
                self.input_layout,
                pointers[1],
                self.weight_layout,
                ct.byref(self.beta),
                pointers[2],
                self.output_layout,
                pointers[2],
                self.output_layout,
                ct.byref(self.result.algorithm),
                workspace.data_ptr(),
                workspace.numel(),
                torch.cuda.current_stream(value.device).cuda_stream,
            )
        )
        return output


class Sm90GdnLinear(CudaF16Linear):
    """Shape dispatch with eager cache refresh and graph-static packed weights."""

    def __init__(self, native):
        super().__init__(native.weight.T, bias=native.bias)
        self._cache_key = None
        self._packed = None
        self._ready = None
        self._pack_event = None
        self._plans = {}
        self._plan_lock = threading.Lock()
        self._refresh_weight()

    def _refresh_weight(self, packed=False):
        weight = self.weight
        if (
            not weight.is_cuda
            or weight.is_inference()
            or weight.requires_grad
            or weight.dtype != torch.bfloat16
        ):
            return False
        key = (
            id(weight),
            weight.data_ptr(),
            weight._version,
            weight.device,
            weight.stride(),
        )
        if key != self._cache_key:
            if torch.cuda.is_current_stream_capturing():
                return False
            self._packed = None
            self._ready = None
            self._pack_event = None
            self._plans.clear()
            self._cache_key = key
        if packed and self._packed is None:
            if torch.cuda.is_current_stream_capturing():
                return False
            self._packed = weight.contiguous()
            # Captured EventWait nodes retain this handle after eager readiness
            # checks stop waiting. Keep it alive with its packed allocation.
            self._pack_event = torch.cuda.Event(external=True)
            self._ready = self._pack_event
            self._ready.record(torch.cuda.current_stream(weight.device))
        return True

    def forward(self, value):
        weight = self.weight
        if not (
            value.ndim == 2
            and value.is_cuda
            and value.device == weight.device
            and value.dtype == weight.dtype == torch.bfloat16
            and self.bias is None
            and value.is_contiguous()
            and not value.requires_grad
            and not weight.requires_grad
            and cached_device_capability(value.device) == (9, 0)
            and torch.cuda.get_device_properties(value.device).multi_processor_count
            == 78
            and tuple(weight.shape) in ((8240, 2560), (2560, 3072))
            and (weight.stride() == (1, weight.shape[0]) or weight.is_contiguous())
            and value.shape[1] == weight.shape[1]
            and value.shape[0] in (16, 32, 64, 8192, 32768)
        ):
            return F.linear(value, weight, self.bias)
        rows = value.shape[0]
        is_input = weight.shape[0] == 8240
        if rows == 32 and not is_input:
            return F.linear(value, weight, self.bias)
        if rows != 8192 and _library() is None:
            return F.linear(value, weight, self.bias)
        packed = is_input and rows <= 64 or not is_input and rows >= 8192
        with self._plan_lock, torch.cuda.device(value.device):
            tracked = self._refresh_weight(packed)
            direct_prefill = (
                not tracked and weight.is_inference() and rows == 8192 and not is_input
            )
            if not tracked and not direct_prefill:
                return F.linear(value, weight, self.bias)
            if tracked and packed and self._ready is not None:
                if torch.cuda.is_current_stream_capturing():
                    torch.cuda.current_stream(value.device).wait_event(self._ready)
                elif self._ready.query():
                    self._ready = None
                else:
                    torch.cuda.current_stream(value.device).wait_event(self._ready)
            # Unversioned weights are consumed directly. Only tracked weights
            # can retain a value-derived packed copy across calls.
            selected = self._packed if tracked and packed else weight
            if packed:
                selected.record_stream(torch.cuda.current_stream(value.device))
            if rows == 8192:
                output = value.new_empty((rows, weight.shape[0]))
                grid = (
                    triton.cdiv(rows, 64) * triton.cdiv(weight.shape[0], 128),
                    1,
                    1,
                )
                key = (rows, packed, value.data_ptr() % 16 == 0)
                launch = self._plans.get(key) if tracked else None
                if launch is None:
                    kernel = _prefill_projection[grid](
                        value,
                        selected,
                        output,
                        weight.shape[0],
                        weight.shape[1],
                        *selected.stride(),
                        num_warps=4,
                        num_stages=3,
                    )
                    if tracked:
                        self._plans[key] = kernel[grid]
                else:
                    launch(
                        value,
                        selected,
                        output,
                        weight.shape[0],
                        weight.shape[1],
                        *selected.stride(),
                        stream=torch.cuda.current_stream(value.device).cuda_stream,
                    )
                return output
            key = (rows, packed)
            if key not in self._plans:
                if torch.cuda.is_current_stream_capturing():
                    return F.linear(value, weight, self.bias)
                self._plans[key] = _MatmulPlan(selected, rows, is_input)
            plan = self._plans[key]
        if not plan.valid:
            return F.linear(value, weight, self.bias)
        with torch.cuda.device(value.device):
            return plan(value, selected)
