import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.perf_test import dsv41_ppu_reference as driver
from rtp_llm.test.perf_test import dsv41_ppu_reference_results as result


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class ReferenceProtocolTest(unittest.TestCase):
    def test_sweep_and_headroom(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(driver.main(["all", "--protocol", "reference-128k", "--dry-run"]), 0)
        cases = json.loads(output.getvalue())
        self.assertEqual([row["point"] for row in cases[:5]], list(result.POINTS["prefill"]))
        self.assertEqual([row["point"] for row in cases[5:]], list(result.POINTS["decode"]))
        self.assertEqual(cases[-1]["global_requests"], 2048)
        self.assertGreaterEqual(cases[-1]["max_seq_len"], result.CONTEXT + result.OUTPUT + 4)
        self.assertEqual(len(driver.prompt(2047).split()), result.CONTEXT - 1)

    def test_prefill_per_device_and_cache_hit_gate(self):
        rows = [dict(
            step=step + 1, start_ns=step * 10**9, end_ns=(step + 1) * 10**9,
            input_min=result.CONTEXT, input_max=result.CONTEXT, batch=2, context_batch=2,
            prefix_min=result.PREFIX, prefix_max=result.PREFIX, fresh_tokens=8192, emitted_tokens=2,
        ) for step in range(15)]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "step-rank0.jsonl"
            write_rows(path, rows)
            metrics = result.prefill_result(directory, 8192)
            self.assertEqual(metrics["steps"], 10)
            self.assertEqual(metrics["step_ms"], 1000)
            self.assertEqual(metrics["throughput_per_ppu"], 1024)
            rows[-1]["prefix_min"] = 0
            write_rows(path, rows)
            with self.assertRaisesRegex(ValueError, "wrong cache hit"):
                result.prefill_result(directory, 8192)

    def test_decode_normalized_and_real_rates(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(result, "OUTPUT", 331):
            directory = Path(temporary)
            for rank in range(result.WORLD):
                rows = [dict(
                    step=step + 1, start_ns=step * 10**8 + rank, end_ns=(step + 1) * 10**8 + rank,
                    input_min=result.CONTEXT, input_max=result.CONTEXT, batch=8, context_batch=0,
                    batch_identity=rank, output_max=(step + 1) * 3, emitted_tokens=24,
                ) for step in range(110)]
                rows[53]["emitted_tokens"] = 0
                rows[54]["emitted_tokens"] = 48
                write_rows(directory / f"step-rank{rank}.jsonl", rows)
                replay = [(row["start_ns"] + 1, mode, 8, mode == "dense")
                          for row in rows for mode in ("proposal", "dense", "commit")]
                write_rows(directory / f"replay-rank{rank}.jsonl", replay)
            metrics = result.decode_result(directory, 8)
            self.assertEqual(metrics["steps"], 60)
            self.assertEqual(metrics["normalized_user_tps"], 32.5)
            self.assertEqual(metrics["throughput_per_ppu"], 260)
            self.assertEqual(metrics["actual_tps_per_ppu"], 240)
            rows[54]["emitted_tokens"] -= 1
            write_rows(directory / "step-rank7.jsonl", rows)
            incomplete = result.decode_result(directory, 8)
            self.assertFalse(incomplete["decode_output_counter_complete"])
            self.assertNotIn("actual_tps_per_ppu", incomplete)
            write_rows(directory / "replay-rank7.jsonl", [entry for entry in replay if entry[1] != "commit"])
            with self.assertRaisesRegex(ValueError, "rank 7 step 51 lacks full"):
                result.decode_result(directory, 8)

    def test_decode_capacity_from_native_pool(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "logs").mkdir()
            log = directory / "logs" / "engine.log"
            self.assertFalse(driver.check_decode_capacity(directory, 8))
            evidence = (
                "[RANK {rank}] cache config: layer_num=40, block_size=3984896B, seq_size_per_block=256\n"
                "[RANK {rank}] CoordinatorCacheManager pool summary: pools=[pool_name=hca_kv, type=FULL, blocks={blocks}]\n"
                "[RANK {rank}] CoordinatorCacheManager set reserve blocks: ratio=5% reserve_blocks=646\n"
            )
            log.write_text(evidence.format(rank=0, blocks=2155) * result.WORLD)
            self.assertFalse(driver.check_decode_capacity(directory, 8))
            log.write_text("".join(evidence.format(rank=rank, blocks=2155) for rank in range(result.WORLD)))
            with self.assertRaisesRegex(RuntimeError, "need 4120 blocks per rank, have 2046"):
                driver.check_decode_capacity(directory, 8)
            self.assertFalse(driver.read(directory / "capacity.json")["admissible"])
            log.write_text("".join(evidence.format(rank=rank, blocks=4700) for rank in range(result.WORLD)))
            self.assertTrue(driver.check_decode_capacity(directory, 8))


if __name__ == "__main__":
    unittest.main()
