import unittest
from unittest.mock import MagicMock, patch

from rtp_llm.test.perf_test.distribution_runner import DistributionRunner
from rtp_llm.test.perf_test.grid_runner import GridRunner


class RunnerLogPathTest(unittest.TestCase):
    def test_grid_and_distribution_forward_log_offsets(self):
        # Exercise real constructors/run loops: mocking the runners themselves
        # would hide a mismatch with the shared kwargs in batch_decode_test.
        for kind in ("grid", "distribution"):
            for log_path in ("", "/tmp/server.log"):
                with self.subTest(kind=kind, log_path=log_path):
                    module = f"rtp_llm.test.perf_test.{kind}_runner"
                    table = (
                        "create_metrics_table"
                        if kind == "grid"
                        else "create_distribution_metrics_table"
                    )
                    with patch(f"{module}.BatchPerfImpl", autospec=True) as impl, patch(
                        f"{module}.{table}"
                    ):
                        batches = [MagicMock(log_flush_offset=n) for n in (0, 17, 42)]
                        impl.side_effect = batches
                        kwargs = dict(
                            dump_json_path="/tmp/results",
                            decode_test_length=10,
                            generate_config={},
                            num_measures=3,
                            log_path=log_path,
                        )
                        if kind == "grid":
                            runner = GridRunner(8000, 1, [1, 2], [128], {128: "q"}, **kwargs)
                        else:
                            runner = DistributionRunner(
                                8000, 1,
                                {"batch_seq_len_map": {"1": [128], "2": [128, 128]}},
                                {128: "q"}, **kwargs,
                            )
                        self.assertEqual(len(runner.run()), 2)
                        self.assertEqual(impl.call_count, 3)
                        self.assertEqual(
                            [c.kwargs["log_path"] for c in impl.call_args_list],
                            [log_path] * 3,
                        )
                        self.assertEqual(
                            [c.kwargs["log_flush_offset"] for c in impl.call_args_list],
                            [0, 0, 17],
                        )
                        for batch in batches[1:]:
                            batch.run.assert_called_once_with(num_measures=3)
                        self.assertEqual(runner._log_flush_offset, 42)


if __name__ == "__main__":
    unittest.main()
