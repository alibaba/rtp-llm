import importlib.util
import os
import unittest
from unittest.mock import patch

import torch


def _load_nan_diag():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.abspath(os.path.join(here, "..", "_nan_diag_triton.py"))
    spec = importlib.util.spec_from_file_location("_dsv4_nan_diag", src)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


nan_diag = _load_nan_diag()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class NanDiagTritonTest(unittest.TestCase):
    def test_detector_is_read_only(self) -> None:
        x = torch.randn((2, 513), dtype=torch.float32, device="cuda")
        x[0, 7] = float("nan")
        x[0, 300] = float("inf")
        before = x.clone()

        with patch.object(nan_diag, "ENABLED", True):
            nan_diag.report_nonfinite(
                x,
                source_id=nan_diag.SOURCE_MOE_INPUT,
                layer_id=17,
            )
            torch.cuda.synchronize()

        torch.testing.assert_close(x, before, equal_nan=True)

    def test_detector_runs_on_every_cuda_graph_replay(self) -> None:
        probe = torch.zeros((1, 256), dtype=torch.bfloat16, device="cuda")
        x = torch.zeros((2, 513), dtype=torch.bfloat16, device="cuda")
        batch_id = torch.tensor([23001], dtype=torch.int64, device="cuda")

        with patch.object(nan_diag, "ENABLED", True):
            nan_diag.set_batch_context(batch_id)
            # JIT must happen outside capture. The live graph shape/strides
            # intentionally differ to verify they do not specialize the kernel.
            nan_diag.report_nonfinite(
                probe,
                source_id=nan_diag.SOURCE_ROUTER_SCORES,
                layer_id=23,
            )
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                nan_diag.report_nonfinite(
                    x,
                    source_id=nan_diag.SOURCE_ROUTER_SCORES,
                    layer_id=23,
                )

            x[1, 300] = float("nan")
            state_index = nan_diag._report_state_index(
                nan_diag.SOURCE_ROUTER_SCORES, 23
            )
            report_count = nan_diag._REPORT_COUNT_BY_DEVICE[str(x.device)]
            before_count = int(report_count[state_index].item())
            for current_batch in (23002, 23003):
                batch_id.fill_(current_batch)
                graph.replay()
                torch.cuda.synchronize()
            self.assertEqual(
                int(report_count[state_index].item()),
                before_count + 2,
            )
            torch.testing.assert_close(
                x[1, 300],
                torch.tensor(float("nan"), dtype=x.dtype, device=x.device),
                equal_nan=True,
            )

    def test_rate_limits_a_nan_storm_per_batch_source_and_layer(self) -> None:
        x = torch.full((8, 1024), float("nan"), dtype=torch.float32, device="cuda")
        batch_id = torch.tensor([991001], dtype=torch.int64, device="cuda")
        source_id = 9  # Test-only source slot.
        layer_id = 997

        with patch.object(nan_diag, "ENABLED", True):
            nan_diag.set_batch_context(batch_id)
            state_index = nan_diag._report_state_index(source_id, layer_id)
            _, report_count = nan_diag._ensure_report_state(x.device)
            before_count = int(report_count[state_index].item())

            nan_diag.report_nonfinite(
                x,
                source_id=source_id,
                layer_id=layer_id,
            )
            torch.cuda.synchronize()
            self.assertEqual(int(report_count[state_index].item()), before_count + 1)

            # Rechecking the same bad tensor in the same model batch is quiet.
            nan_diag.report_nonfinite(
                x,
                source_id=source_id,
                layer_id=layer_id,
            )
            torch.cuda.synchronize()
            self.assertEqual(int(report_count[state_index].item()), before_count + 1)

            # A new model batch must produce a new event.
            batch_id.fill_(991002)
            nan_diag.report_nonfinite(
                x,
                source_id=source_id,
                layer_id=layer_id,
            )
            torch.cuda.synchronize()
            self.assertEqual(int(report_count[state_index].item()), before_count + 2)

    def test_attention_lse_ignores_negative_inf_but_reports_nan(self) -> None:
        lse = torch.zeros((2, 3, 7), dtype=torch.float32, device="cuda")
        lse[0, 0, 0] = -float("inf")
        batch_id = torch.tensor([992001], dtype=torch.int64, device="cuda")

        with patch.object(nan_diag, "ENABLED", True):
            nan_diag.set_batch_context(batch_id)
            state_index = nan_diag._report_state_index(
                nan_diag.SOURCE_CP_ATTENTION_LSE, 12
            )
            _, report_count = nan_diag._ensure_report_state(lse.device)
            before_count = int(report_count[state_index].item())

            nan_diag.report_nonfinite(
                lse,
                source_id=nan_diag.SOURCE_CP_ATTENTION_LSE,
                layer_id=12,
                include_neg_inf=False,
            )
            torch.cuda.synchronize()
            self.assertEqual(int(report_count[state_index].item()), before_count)

            lse[1, 2, 6] = float("nan")
            nan_diag.report_nonfinite(
                lse,
                source_id=nan_diag.SOURCE_CP_ATTENTION_LSE,
                layer_id=12,
                include_neg_inf=False,
            )
            torch.cuda.synchronize()
            self.assertEqual(int(report_count[state_index].item()), before_count + 1)


if __name__ == "__main__":
    unittest.main()
