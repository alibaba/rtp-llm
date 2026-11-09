"""SM100 numerical and dynamic CUDA Graph validation, without model weights."""

import ctypes
import importlib.util
import json
import os
import pathlib
import sys
import time
from types import SimpleNamespace

root = pathlib.Path(os.environ["GLM53_EXTENSION_TEST_ROOT"])
for key, name in {
    "TMPDIR": "tmp",
    "TMP": "tmp",
    "TEMP": "tmp",
    "XDG_CACHE_HOME": "cache",
    "DG_JIT_CACHE_DIR": "cache/deepgemm",
    "FLASHINFER_WORKSPACE_BASE": "cache/flashinfer",
    "FLASHINFER_JIT_DIR": "cache/flashinfer/jit",
    "TILELANG_CACHE_DIR": "cache/tilelang",
    "TRITON_CACHE_DIR": "cache/triton",
    "CUDA_CACHE_PATH": "cache/cuda",
}.items():
    p = root / name
    p.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(p)

import torch

torch.set_num_threads(1)
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
assert torch.cuda.get_device_capability() == (10, 0)
assert torch.cuda.mem_get_info()[0] > 4 * 1024**3
torch.manual_seed(20261105)
package = pathlib.Path(importlib.util.find_spec("rtp_llm").origin).parent
assets = pathlib.Path(
    os.environ.get("GLM53_EXTENSION_ASSET_ROOT", str(package / "libs"))
)
result = {
    "started": time.time(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "capability": torch.cuda.get_device_capability(),
    "sms": torch.cuda.get_device_properties(0).multi_processor_count,
    "rows": [],
}


def record(row):
    result["rows"].append(row)
    (root / "validation_result.json").write_text(json.dumps(result, indent=2))
    print("VALIDATION", json.dumps(row), flush=True)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def sparse():
    os.environ["GLM53_NATIVE_SPARSE_LIBRARY"] = str(
        assets / "flash_mla_glm53_22.abi3.so"
    )
    native_module = module(
        "codex_glm53_sparse",
        package
        / "models_py/modules/factory/attention/cuda_mla_impl/glm53_native_sparse.py",
    )
    batch, ctx, heads, width, topk = 48, 81920, 64, 512, 2176
    cache = torch.empty(
        (batch * ctx // 128, 128, 528), device="cuda", dtype=torch.uint8
    )
    flat = cache.view(-1, 528)
    for start in range(0, batch * ctx, 4096):
        n = min(4096, batch * ctx - start)
        x = torch.randn(n, 4, 128, device="cuda", dtype=torch.bfloat16) * 0.5
        scales = x.float().abs().amax(-1).clamp_min(1e-4) / 448
        raw = (x.float() / scales[..., None]).reshape(n, 512).to(torch.float8_e4m3fn)
        flat[start : start + n, :512].copy_(raw.view(torch.uint8))
        flat[start : start + n, 512:].copy_(scales.contiguous().view(torch.uint8))
    del x, raw, scales

    def reference(q, indices, actual):
        largest = 0.0
        for start in range(0, q.shape[0], 4):
            ids = indices[start : start + 4]
            selected = flat[ids.clamp_min(0).long()]
            values = (
                selected[..., :512]
                .contiguous()
                .view(torch.float8_e4m3fn)
                .float()
                .reshape(-1, topk, 4, 128)
            )
            scales = selected[..., 512:].contiguous().view(torch.float32)
            kv = (
                (values * scales[..., None])
                .reshape(-1, topk, 512)
                .to(torch.bfloat16)
                .float()
            )
            logits = torch.bmm(q[start : start + 4].float(), kv.transpose(1, 2)) * (
                width**-0.5
            )
            logits.masked_fill_(ids[:, None, :] < 0, float("-inf"))
            prob = torch.nan_to_num(logits.softmax(-1), nan=0.0)
            target = torch.bmm(prob, kv).to(torch.bfloat16)
            torch.testing.assert_close(
                actual[start : start + 4], target, rtol=0.02, atol=0.002
            )
            largest = max(
                largest,
                (actual[start : start + 4].float() - target.float()).abs().max().item(),
            )
        return largest

    for invariant in (False, True):
        native_module._INVARIANT = invariant
        for tokens in (1, 4, 5, 6):
            wrapper = native_module.Glm53NativeSparse()
            assert wrapper.native is not None, "Native sparse extension did not load"
            rows = batch * tokens
            q = (
                torch.randn(rows, heads, width, device="cuda", dtype=torch.bfloat16)
                * 0.3
            )
            ids = torch.randint(
                0, ctx, (batch, tokens, topk), device="cuda", dtype=torch.int32
            )
            ids += (
                torch.arange(batch, device="cuda", dtype=torch.int32)[:, None, None]
                * ctx
            )
            ids[:, :, 2051:] = -1
            ids[0, :, 0:3] = torch.tensor(
                [127, 128, ctx - 1], device="cuda", dtype=torch.int32
            )
            ids = ids.reshape(rows, topk)
            original = ids.clone()

            def operation():
                return wrapper.forward(
                    q, cache, ids, width**-0.5, tokens_per_request=tokens
                )

            actual = operation()
            assert actual is not None
            err = reference(q, ids, actual)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    operation()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = operation()
            torch.cuda.current_stream().wait_stream(stream)
            checks = []
            for case in ("holes", "empty_request", "new_query"):
                ids.copy_(original)
                if case == "holes":
                    ids[:, ::7] = -1
                elif case == "empty_request":
                    ids[:tokens].fill_(-1)
                else:
                    q.mul_(0.91)
                graph.replay()
                torch.cuda.synchronize()
                checks.append({"case": case, "max_abs": reference(q, ids, captured)})
            record(
                {
                    "extension": "native_sparse",
                    "tokens": tokens,
                    "batch": batch,
                    "context": ctx,
                    "batch_invariant": invariant,
                    "max_abs": err,
                    "graph_checks": checks,
                    "status": "passed",
                }
            )
    torch.cuda.empty_cache()


def mhc():
    os.environ["GLM53_MEGA_MHC_LIBRARY_DIR"] = str(assets / "mega_mhc_isolated")
    sys.path.insert(0, str(package / "models_py/3rdparty"))
    try:
        import z3

        ctypes.CDLL(
            str(pathlib.Path(z3.__file__).parent / "lib/libz3.so"),
            mode=ctypes.RTLD_GLOBAL,
        )
    except (ImportError, OSError):
        pass
    from flashinfer.norm import rmsnorm
    from tile_kernels.modeling.mhc.functional import mhc_post
    from tile_kernels.modeling.mhc.ops.pre_big_fuse import mhc_pre_big_fuse

    m = module("codex_glm53_mhc", package / "models_py/modules/glm53_mega_mhc.py")
    for rows in (48, 192, 240, 288):
        x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16) * 0.2
        residual = torch.randn(rows, 4, 4096, device="cuda", dtype=torch.bfloat16) * 0.2
        post = torch.randn(rows, 4, 1, device="cuda").sigmoid()
        comb = torch.randn(rows, 4, 4, device="cuda").softmax(-1)
        fn = torch.randn(24, 16384, device="cuda") * 0.001
        unit = SimpleNamespace(
            hc_mult=4,
            dim=4096,
            fn=fn,
            scale=torch.ones(3, device="cuda"),
            base=torch.zeros(24, device="cuda"),
            norm_eps=1e-6,
            hc_eps=1e-6,
            hc_sinkhorn_iters=10,
        )
        norm = SimpleNamespace(
            weight=torch.randn(4096, device="cuda", dtype=torch.bfloat16) * 0.1 + 1,
            variance_epsilon=1e-6,
        )
        assert m.can_use_mega_mhc(
            x, residual, post, comb, unit, norm
        ), "MegaMHC fast path rejected supported shape"

        def reference():
            updated = mhc_post(x, residual, post, comb)
            np, nc, y = mhc_pre_big_fuse(
                updated,
                fn,
                unit.scale,
                unit.base,
                rms_eps=1e-6,
                mhc_pre_eps=1e-6,
                mhc_sinkhorn_eps=1e-6,
                mhc_post_mult_value=2.0,
                sinkhorn_repeat=10,
            )
            return updated, rmsnorm(y, norm.weight, eps=1e-6), np, nc

        def operation():
            return m.fused_post_pre_rms(x, residual, post, comb, unit, norm)

        target = reference()
        actual = operation()
        torch.cuda.synchronize()
        for a, b in zip(actual, target):
            torch.testing.assert_close(a, b, rtol=0.01, atol=0.005)
        # Warm only the execution stream. First use of a distinct capture
        # stream must take an independent, already-zeroed barrier.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = operation()
        torch.cuda.current_stream().wait_stream(stream)
        x.mul_(0.97)
        residual.mul_(0.93)
        target = reference()
        graph.replay()
        torch.cuda.synchronize()
        for a, b in zip(captured, target):
            torch.testing.assert_close(a, b, rtol=0.01, atol=0.005)
        record(
            {
                "extension": "mega_mhc",
                "rows": rows,
                "max_abs": [
                    (a.float() - b.float()).abs().max().item()
                    for a, b in zip(captured, target)
                ],
                "dynamic_capture_stream": "passed",
                "status": "passed",
            }
        )


try:
    sparse()
    mhc()
except Exception as error:
    result["error"] = repr(error)
    raise
finally:
    result["finished"] = time.time()
    result["passed"] = len(result["rows"]) == 12 and all(
        r["status"] == "passed" for r in result["rows"]
    )
    (root / "validation_result.json").write_text(json.dumps(result, indent=2))
    print(
        "VALIDATION_COMPLETE",
        json.dumps({k: v for k, v in result.items() if k != "rows"}),
        flush=True,
    )
