"""Isolated cuBLASLt algorithm search against the frozen raw-X projection."""

import ctypes as ct
import json
import os
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv4.fp8.compressor import _linear_bf16_bf16_fp32


class Algo(ct.Structure):
    _fields_ = [("data", ct.c_uint64 * 8)]


class Result(ct.Structure):
    _fields_ = [
        ("algo", Algo),
        ("workspace", ct.c_size_t),
        ("state", ct.c_int),
        ("waves", ct.c_float),
        ("reserved", ct.c_int * 4),
    ]


class Lt:
    def __init__(self):
        self.lib = ct.CDLL("libcublasLt.so.13")
        self.lib.cublasLtGetVersion.restype = ct.c_size_t
        self.version = self.lib.cublasLtGetVersion()
        signatures = {
            "Create": [ct.POINTER(ct.c_void_p)],
            "Destroy": [ct.c_void_p],
            "MatmulDescCreate": [ct.POINTER(ct.c_void_p), ct.c_int, ct.c_int],
            "MatmulDescDestroy": [ct.c_void_p],
            "MatmulDescSetAttribute": [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_size_t],
            "MatrixLayoutCreate": [
                ct.POINTER(ct.c_void_p),
                ct.c_int,
                ct.c_uint64,
                ct.c_uint64,
                ct.c_int64,
            ],
            "MatrixLayoutDestroy": [ct.c_void_p],
            "MatmulPreferenceCreate": [ct.POINTER(ct.c_void_p)],
            "MatmulPreferenceDestroy": [ct.c_void_p],
            "MatmulPreferenceSetAttribute": [
                ct.c_void_p,
                ct.c_int,
                ct.c_void_p,
                ct.c_size_t,
            ],
            "MatmulAlgoGetHeuristic": [ct.c_void_p] * 7
            + [ct.c_int, ct.POINTER(Result), ct.POINTER(ct.c_int)],
            "MatmulAlgoConfigGetAttribute": [
                ct.POINTER(Algo),
                ct.c_int,
                ct.c_void_p,
                ct.c_size_t,
                ct.POINTER(ct.c_size_t),
            ],
            "MatmulAlgoConfigSetAttribute": [
                ct.POINTER(Algo),
                ct.c_int,
                ct.c_void_p,
                ct.c_size_t,
            ],
            "MatmulAlgoCheck": [ct.c_void_p] * 6
            + [ct.POINTER(Algo), ct.POINTER(Result)],
            "Matmul": [ct.c_void_p] * 12
            + [ct.POINTER(Algo), ct.c_void_p, ct.c_size_t, ct.c_void_p],
        }
        for name, types in signatures.items():
            fn = getattr(self.lib, "cublasLt" + name)
            fn.argtypes, fn.restype = types, ct.c_int
        self.handle = ct.c_void_p()
        self.check(self.lib.cublasLtCreate(ct.byref(self.handle)))
        self.desc = ct.c_void_p()
        self.check(self.lib.cublasLtMatmulDescCreate(ct.byref(self.desc), 68, 0))
        trans = ct.c_int(1)
        self.check(
            self.lib.cublasLtMatmulDescSetAttribute(
                self.desc, 3, ct.byref(trans), ct.sizeof(trans)
            )
        )
        self.preference = ct.c_void_p()
        self.check(self.lib.cublasLtMatmulPreferenceCreate(ct.byref(self.preference)))
        limit = ct.c_size_t(32 * 2**20)
        self.check(
            self.lib.cublasLtMatmulPreferenceSetAttribute(
                self.preference, 1, ct.byref(limit), ct.sizeof(limit)
            )
        )
        self.workspace = torch.empty(limit.value, device="cuda", dtype=torch.uint8)
        self.layouts = {}

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError(f"cuBLASLt status {status}")

    def layout(self, rows):
        if rows not in self.layouts:
            result = []
            for dtype, m, n, ld in (
                (14, 5120, 512, 5120),
                (14, 5120, rows, 5120),
                (0, 512, rows, 512),
            ):
                descriptor = ct.c_void_p()
                self.check(
                    self.lib.cublasLtMatrixLayoutCreate(
                        ct.byref(descriptor), dtype, m, n, ld
                    )
                )
                result.append(descriptor)
            self.layouts[rows] = result
        return self.layouts[rows]

    def heuristics(self, rows):
        a, b, c = self.layout(rows)
        results, count = (Result * 64)(), ct.c_int()
        self.check(
            self.lib.cublasLtMatmulAlgoGetHeuristic(
                self.handle,
                self.desc,
                a,
                b,
                c,
                c,
                self.preference,
                64,
                results,
                ct.byref(count),
            )
        )
        return [
            Algo.from_buffer_copy(results[i].algo)
            for i in range(count.value)
            if results[i].state == 0
        ]

    def config(self, algo):
        result = {}
        for attr in range(9):
            value, written = ct.c_int(), ct.c_size_t()
            status = self.lib.cublasLtMatmulAlgoConfigGetAttribute(
                ct.byref(algo),
                attr,
                ct.byref(value),
                ct.sizeof(value),
                ct.byref(written),
            )
            if status == 0:
                result[str(attr)] = value.value
        return result

    def call(self, x, weight, algo):
        a, b, c = self.layout(len(x))
        check = Result()
        status = self.lib.cublasLtMatmulAlgoCheck(
            self.handle, self.desc, a, b, c, c, ct.byref(algo), ct.byref(check)
        )
        if status or check.state or check.workspace > self.workspace.numel():
            return None
        out = torch.empty((len(x), 512), dtype=torch.float32, device=x.device)
        alpha, beta = ct.c_float(1), ct.c_float(0)
        status = self.lib.cublasLtMatmul(
            self.handle,
            self.desc,
            ct.byref(alpha),
            weight.data_ptr(),
            a,
            x.data_ptr(),
            b,
            ct.byref(beta),
            out.data_ptr(),
            c,
            out.data_ptr(),
            c,
            ct.byref(algo),
            self.workspace.data_ptr(),
            self.workspace.numel(),
            torch.cuda.current_stream().cuda_stream,
        )
        if status:
            return None
        return out, check.workspace

    def close(self):
        torch.cuda.synchronize()
        for layouts in self.layouts.values():
            for layout in layouts:
                self.check(self.lib.cublasLtMatrixLayoutDestroy(layout))
        self.check(self.lib.cublasLtMatmulPreferenceDestroy(self.preference))
        self.check(self.lib.cublasLtMatmulDescDestroy(self.desc))
        self.check(self.lib.cublasLtDestroy(self.handle))


class CompactCPLt(unittest.TestCase):
    def test_fixed_algorithm_search(self):
        root = Path(os.environ["DSV41_CP_FIXTURES"])
        fixture = torch.load(
            root / "layer2-chunk0.pt", map_location="cuda", weights_only=True
        )
        weights = torch.load(
            root / "weights-layer2.pt", map_location="cuda", weights_only=True
        )
        x, weight = fixture["tensors"]["x"], weights["global_wkv"]
        expected = _linear_bf16_bf16_fp32(x, weight)
        lt = Lt()
        report = {
            "candidates": [],
            "exact": [],
            "cublas_lt_version": lt.version,
            "loaded_libraries": sorted(
                {
                    line.split()[-1]
                    for line in Path("/proc/self/maps").read_text().splitlines()
                    if "libcublas" in line
                }
            ),
        }
        try:
            algorithms = {}
            for rows in (4096, 1024, 512):
                for algo in lt.heuristics(rows):
                    algorithms[bytes(algo)] = algo
            for raw, algo in algorithms.items():
                output = lt.call(x[:1024], weight, algo)
                if output is None:
                    continue
                actual, workspace = output
                same = torch.equal(expected[:1024], actual)
                entry = {
                    "config": lt.config(algo),
                    "algo_hex": raw.hex(),
                    "workspace": workspace,
                    "exact": same,
                    "max_abs_difference": float((expected[:1024] - actual).abs().max()),
                }
                report["candidates"].append(entry)
                if same:
                    report["exact"].append(entry)
            from test_v41_compact_cp_arithmetic_sm120 import difference, produce

            report["qualified"] = []
            for candidate in report["exact"]:
                algo = Algo.from_buffer_copy(bytes.fromhex(candidate["algo_hex"]))
                qualified = {
                    "config": candidate["config"],
                    "checks": [],
                    "passed": True,
                }
                for path in sorted(root.glob("layer*-chunk*.pt")):
                    fixture = torch.load(path, map_location="cuda", weights_only=True)
                    fixture.update(fixture.pop("tensors"))
                    weights = torch.load(
                        root / f'weights-layer{fixture["layer"]}.pt',
                        map_location="cuda",
                        weights_only=True,
                    )
                    x = fixture["x"]
                    expected_values = _linear_bf16_bf16_fp32(x, weights["global_wkv"])
                    expected_scores = (
                        _linear_bf16_bf16_fp32(x, weights["global_wgate"])
                        if fixture["ratio"] == 2
                        else None
                    )
                    baseline = produce(
                        expected_values, expected_scores, weights, fixture
                    )
                    per_rank = []
                    for rank in range(4):
                        indices = torch.cat(
                            (
                                torch.arange(
                                    rank * 512, (rank + 1) * 512, device=x.device
                                ),
                                torch.arange(
                                    (7 - rank) * 512, (8 - rank) * 512, device=x.device
                                ),
                            )
                        )
                        local = x.index_select(0, indices)
                        values = lt.call(local, weights["global_wkv"], algo)[0]
                        checks = {
                            "values": difference(expected_values[indices], values)
                        }
                        scores = None
                        if expected_scores is not None:
                            scores = lt.call(local, weights["global_wgate"], algo)[0]
                            checks["scores"] = difference(
                                expected_scores[indices], scores
                            )
                        local_outputs = []
                        for half in (0, 512):
                            first = rank * 512 if half == 0 else (7 - rank) * 512
                            info = dict(fixture)
                            info["freqs"] = fixture["freqs"][
                                first
                                // fixture["ratio"] : (first + 512)
                                // fixture["ratio"]
                            ]
                            local_outputs.append(
                                produce(
                                    values[half : half + 512],
                                    (
                                        scores[half : half + 512]
                                        if scores is not None
                                        else None
                                    ),
                                    weights,
                                    info,
                                )
                            )
                        per_rank.append(local_outputs)
                        qualified["checks"].append(
                            {"fixture": path.name, "rank": rank, "boundaries": checks}
                        )
                        qualified["passed"] &= all(v["exact"] for v in checks.values())
                    pieces = [per_rank[r][0] for r in range(4)] + [
                        per_rank[r][1] for r in reversed(range(4))
                    ]
                    checks = {
                        name: difference(
                            value, torch.cat([piece[name] for piece in pieces])
                        )
                        for name, value in baseline.items()
                    }
                    qualified["checks"].append(
                        {"fixture": path.name, "rank": "restored", "boundaries": checks}
                    )
                    qualified["passed"] &= all(v["exact"] for v in checks.values())
                report["qualified"].append(qualified)
            Path(os.environ["DSV41_CP_PROBE_OUTPUT"]).write_text(
                json.dumps(report, indent=2) + "\n"
            )
            self.assertTrue(
                any(c["passed"] for c in report["qualified"]),
                "No cuBLASLt algorithm qualified across all captures",
            )
        finally:
            lt.close()


if __name__ == "__main__":
    unittest.main()
