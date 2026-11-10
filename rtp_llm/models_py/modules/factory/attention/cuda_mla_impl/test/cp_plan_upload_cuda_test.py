import json
import os
import statistics
import time
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.cp_kv_prefetch import (
    _upload_immutable_plan,
    prefix_plan,
)


class CpPlanUploadCudaTest(unittest.TestCase):
    def test_upload_values_transfer_count_and_host_time(self):
        self.assertTrue(torch.cuda.is_available(), "requires an allocated CUDA GPU")
        root = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        results = []
        for name, prefixes, lengths in (
            ("small", [4096], [82]),
            ("large", [32768, 65536, 131072], [29768, 20280, 4906]),
        ):
            base_plan = prefix_plan(prefixes, lengths, 8, 0, 64)

            def new_row_metadata(device: str):
                return {
                    "new_lengths": torch.tensor(
                        [len(base_plan["new_local_rows"])],
                        dtype=torch.int32,
                        device=device,
                    ),
                    "new_starts": torch.zeros(1, dtype=torch.int32, device=device),
                    "new_padded_lengths": torch.tensor(
                        [base_plan["new_max"]], dtype=torch.int32, device=device
                    ),
                }

            plan = {**base_plan, **new_row_metadata("cpu")}
            expected = {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in plan.items()
            }
            tensor_count = sum(torch.is_tensor(value) for value in plan.values())
            dtype_count = len(
                {value.dtype for value in plan.values() if torch.is_tensor(value)}
            )

            def upload(batched: bool):
                if batched:
                    return _upload_immutable_plan(
                        {**base_plan, **new_row_metadata("cpu")}, "cuda"
                    )
                uploaded = {
                    key: value.to("cuda") if torch.is_tensor(value) else value
                    for key, value in base_plan.items()
                }
                uploaded.update(new_row_metadata("cuda"))
                return uploaded, []

            for batched in (False, True):
                mode = "batched" if batched else "individual"
                for _ in range(3):
                    uploaded, owners = upload(batched)
                    torch.cuda.synchronize()
                samples = []
                for _ in range(30):
                    torch.cuda.synchronize()
                    start = time.perf_counter_ns()
                    uploaded, owners = upload(batched)
                    samples.append((time.perf_counter_ns() - start) / 1000)
                    torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profiler:
                    uploaded, owners = upload(batched)
                    torch.cuda.synchronize()
                trace_path = root / f"cp_plan_upload_{name}_{mode}.json"
                profiler.export_chrome_trace(str(trace_path))
                transfers = [
                    event
                    for event in profiler.events()
                    if event.device_type == torch.autograd.DeviceType.CUDA
                    and "Memcpy HtoD" in event.name
                ]
                self.assertEqual(
                    len(transfers), dtype_count if batched else tensor_count - 1
                )
                for key, value in expected.items():
                    if torch.is_tensor(value):
                        self.assertTrue(torch.equal(uploaded[key].cpu(), value), key)
                    else:
                        self.assertEqual(uploaded[key], value)
                self.assertEqual(len(owners), dtype_count if batched else 0)
                results.append(
                    {
                        "case": name,
                        "mode": mode,
                        "cp_size": 8,
                        "prefixes": prefixes,
                        "lengths": lengths,
                        "h2d_calls": len(transfers),
                        "host_submit_p50_us": statistics.median(samples),
                        "host_submit_p95_us": sorted(samples)[28],
                        "host_submit_samples_us": samples,
                        "values_exact": True,
                        "trace": str(trace_path),
                    }
                )
        (root / "cp_plan_upload_benchmark.json").write_text(
            json.dumps(results, indent=2)
        )
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    unittest.main()
