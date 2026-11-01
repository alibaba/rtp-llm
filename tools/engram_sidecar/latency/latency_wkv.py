"""Real DeepGEMM WKV stage for the isolated four-path Engram benchmark.

Initialization, scale packing, allocation, JIT warmup and comparisons are outside
timing. ``WKVRunner.__call__`` launches the same DeepGEMM API/recipe as main
V41MXFP8Linear.forward_quantized into preallocated BF16 output. This is an
experimental fixed synthetic WKV, not a mock of compute_ops or a full model.

Usage (all four inputs must have identical geometry)::

    runners, metadata = prepare_wkv_paths(
        paths.keys(), batch=M, k=H * D, n=256, device='cuda:0')
    # Each timed path: original lookup+quant OR canonical gather, then:
    runners[name](out_q, out_s)
    # Warm up every full path before capture, as run_bench already does.
    # After eager/replay completion, outside timing:
    verification = compare_wkv_outputs(runners, baseline='local_raw')

Each path has its own output; all paths share exactly the same fixed local HBM
weight allocation. Retain runners/weights until GPU drain AND Graph reset.
No manual close is exposed because the caller owns that lifecycle boundary.
"""

from dataclasses import dataclass
import hashlib
import importlib.metadata
import math

import torch


@dataclass(frozen=True)
class _Weights:
    q: torch.Tensor
    packed_scales: torch.Tensor
    block_scale_bytes_cpu: torch.Tensor
    metadata: dict


def _device(value):
    device = torch.device('cuda', value) if type(value) is int else torch.device(value)
    if device.type != 'cuda' or device.index is None:
        raise ValueError('an explicit CUDA device index is required')
    return device


class WKVRunner:
    """One path's output scratch over shared read-only synthetic WKV weights.

    First use can JIT compile inside DeepGEMM: warm up before Graph capture and
    timing. Subsequent calls contain metadata validation and an asynchronous
    GEMM; the wrapper itself does not allocate torch tensors, copy data or
    synchronize CUDA. DeepGEMM's implementation remains the actual backend.
    """

    def __init__(self, weights, *, batch, device):
        if type(batch) is not int or batch < 0:
            raise ValueError('batch must be a nonnegative integer')
        self.weights = weights
        self.device = _device(device)
        self.batch = batch
        self.n, self.k = weights.q.shape
        if weights.q.device != self.device or weights.packed_scales.device != self.device:
            raise ValueError('weight device mismatch')
        import deep_gemm
        self._gemm = deep_gemm.fp8_fp4_gemm_nt
        with torch.cuda.device(self.device):
            self.output = torch.empty((batch, self.n), device=self.device, dtype=torch.bfloat16)

    def __call__(self, q, scales):
        if (q.device != self.device or scales.device != self.device
                or q.dtype != torch.float8_e4m3fn or scales.dtype != torch.int32
                or q.shape != (self.batch, self.k) or not q.is_contiguous()
                or scales.shape != (self.batch, (self.k + 127) // 128)
                or scales.stride() != (1, (self.batch + 3) // 4 * 4)):
            raise ValueError('WKV input must be contiguous E4M3 [M,K] plus packed/TMA-strided int32 scales')
        if self.batch:
            with torch.cuda.device(self.device):
                self._gemm((q, scales), (self.weights.q, self.weights.packed_scales),
                           self.output, recipe=(1, 1, 32))
        return self.output


def prepare_wkv_paths(names, *, batch, k, n=256, device='cuda:0', seed=4101610):
    """Create one fixed weight dataset and independent outputs for each path.

    Conservative experimental geometry: N multiple128, K multiple256, reflecting
    the tested SM100 MXFP8 configurations. No padding or changed algorithm is
    silently inserted for unsupported geometry. All preparation is untimed.
    """
    names = list(names)
    if not names or len(set(names)) != len(names) or any(not isinstance(x, str) or not x for x in names):
        raise ValueError('distinct nonempty path names required')
    if type(k) is not int or k <= 0 or k % 256 or type(n) is not int or n <= 0 or n % 128:
        raise ValueError('experimental WKV requires K a positive multiple256 and N a positive multiple128')
    if type(seed) is not int or type(batch) is not int or batch < 0:
        raise ValueError('integer seed and nonnegative integer batch required')
    device = _device(device)
    from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor
    gen = torch.Generator(device='cpu').manual_seed(seed)
    # Same storage convention as checkpoint: FP8 [N,K] and E8M0 [N/32,K/32].
    q_cpu = (torch.randn((n, k), generator=gen) * 2).to(torch.float8_e4m3fn)
    scale_bytes = torch.randint(123, 128, (n // 32, k // 32), generator=gen).to(torch.uint8)
    with torch.cuda.device(device):
        weight = q_cpu.to(device)
        checkpoint_scales = scale_bytes.to(device).view(torch.float8_e8m0fnu)
        rows = torch.arange(n, device=device) // 32
        # Exact V41MXFP8Linear.pack_scales expansion/layout helper.
        packed = get_mn_major_tma_aligned_packed_ue8m0_tensor(
            checkpoint_scales.float().index_select(0, rows))
    metadata = dict(
        experiment='fixed-synthetic-wkv-v1', seed=seed, n=n, k=k, batch=batch,
        weight_dtype='float8_e4m3fn', checkpoint_scale_dtype='float8_e8m0fnu',
        checkpoint_scale_shape=list(scale_bytes.shape), checkpoint_block=[32, 32],
        weight_sha256=hashlib.sha256(q_cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
        scale_bytes_sha256=hashlib.sha256(scale_bytes.numpy().tobytes()).hexdigest(),
        gemm='deep_gemm.fp8_fp4_gemm_nt', recipe=[1, 1, 32],
        deep_gemm=importlib.metadata.version('deep-gemm'), torch=torch.__version__,
        output_dtype='bfloat16', shared_weights_across_paths=True,
        independent_preallocated_outputs=True, warmup_required_before_capture=True,
        weight_preparation_timed=False, comparison_timed=False,
        scope='synthetic WKV component after original quant or canonical gather; not full model',
    )
    weights = _Weights(weight, packed, scale_bytes, metadata)
    runners = {name: WKVRunner(weights, batch=batch, device=device) for name in names}
    return runners, dict(metadata)


def compare_wkv_outputs(runners, *, baseline='local_raw', rtol=0.0, atol=0.0):
    """Untimed D2H synchronization and four-path numerical comparison.

    Invoke while all outputs represent the same IDs. By default require exact
    BF16 equality, since identical q/scale bytes feed the same GEMM operation.
    Returns public JSON metadata, never handles or tensor contents.
    """
    if baseline not in runners or not all(isinstance(x, WKVRunner) for x in runners.values()):
        raise ValueError('known baseline and WKVRunner paths required')
    if not all(math.isfinite(x) and x >= 0 for x in (rtol, atol)):
        raise ValueError('finite nonnegative comparison tolerances required')
    weights = runners[baseline].weights
    if any(r.weights is not weights for r in runners.values()):
        raise ValueError('benchmark paths must share the same fixed weight object')
    reference = runners[baseline].output.float().cpu()
    if not bool(torch.isfinite(reference).all()):
        raise AssertionError('nonfinite WKV baseline output')
    records = {}
    for name, runner in runners.items():
        actual = runner.output.float().cpu()
        torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)
        records[name] = dict(elements=actual.numel(),
                             max_abs_error=float((actual-reference).abs().max()) if actual.numel() else 0.0)
    return dict(status='PASS', baseline=baseline, rtol=rtol, atol=atol,
                comparisons=records, scope='same IDs and same synthetic WKV weight')


def compare_wkv_cpu_reference(runner, q, scales, *, rtol=0.01, atol=0.125):
    """Optional untimed independent CPU dequant+matmul check for small fixtures.

    Copies the selected quantized activations and fixed small synthetic weights.
    Never call inside capture/timing. This does not load/copy an Engram table.
    """
    if not isinstance(runner, WKVRunner):
        raise TypeError('WKVRunner required')
    if runner.batch == 0:
        if q.shape != (0, runner.k) or scales.shape != (0, (runner.k + 127)//128):
            raise ValueError('empty input geometry mismatch')
        return dict(status='PASS', elements=0, rtol=rtol, atol=atol,
                    max_abs_error=0.0, scope='empty WKV has no GEMM launch')
    words = scales.cpu().to(torch.int64) & 0xffffffff
    exponents = torch.stack([(words >> (8*i)) & 255 for i in range(4)], -1)
    exponents = exponents.reshape(runner.batch, -1)[:, :runner.k // 32].float()
    activations = q.float().cpu() * torch.pow(2., exponents-127).repeat_interleave(32, 1)
    blocks = torch.pow(2., runner.weights.block_scale_bytes_cpu.float()-127)
    weights = runner.weights.q.float().cpu() * blocks.repeat_interleave(32,0).repeat_interleave(32,1)
    reference = (activations @ weights.T).to(torch.bfloat16).float()
    actual = runner.output.float().cpu()
    torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)
    return dict(status='PASS', elements=actual.numel(), rtol=rtol, atol=atol,
                max_abs_error=float((actual-reference).abs().max()) if actual.numel() else 0.0,
                scope='CPU dequantized dot product vs synthetic WKV component')
