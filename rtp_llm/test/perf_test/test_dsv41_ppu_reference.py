import asyncio
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.test.perf_test import dsv41_ppu_reference as driver
from rtp_llm.test.perf_test import dsv41_ppu_reference_results as result
from rtp_llm.test.perf_test import dsv41_ppu_reference_profile as profiler


class ProfileAdmissionTest(unittest.IsolatedAsyncioTestCase):
    async def test_dp_profile_is_bound_to_the_eight_actual_requests(self):
        events = []
        payloads = []

        class Client:
            async def post(self, url, **kwargs):
                events.append((url.rsplit("/", 1)[-1], kwargs.get("json")))
                if url.endswith("/batch_infer"):
                    payloads.append(json.loads(kwargs["content"]))
                return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"status": "ok"})

        async def load(config):
            client = Client()
            for wave in range(1, 16):
                await client.post("http://owner0/update_scheduler_info", json={"batch_size": 2})
                pending = [asyncio.create_task(client.post(f"http://owner{rank}/batch_infer", content=json.dumps({
                    "prompt_batch": [f"wave-{wave}-rank-{rank}"], "generate_config": {"max_new_tokens": 1},
                }).encode()))
                           for rank in range(8)]
                await asyncio.sleep(0)
                await client.post("http://owner0/update_scheduler_info", json={"batch_size": 1})
                await asyncio.gather(*pending)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for rank in range(8):
                (directory / f"prefill_wr{rank}_1.json").write_text(json.dumps({
                    "traceEvents": [{"cat": "kernel"}, {"cat": "cpu_op", "name":
                        "executor.model_forward(ctx_batch=1,gen_batch=0,tokens=4096,max_seq=131072)"}],
                }))
            config = {
                "case": driver.case_config("prefill", 4096, "dp8-compact140"),
                "node": {"profile_warmup_wave": 5, "profile_trace_name": "prefill", "environment": {
                    "TORCH_CUDA_PROFILER_DIR": temporary, driver.DP_PREQUEUE_ENV: "1500",
                }}, "prefill_address": "owner0:28100", "output": temporary,
            }
            with patch.dict("sys.modules", httpx=SimpleNamespace(AsyncClient=Client)), patch.object(driver, "load", load):
                await profiler.load(config)
            positions = [index for index, (name, _) in enumerate(events) if name == "start_profile"]
            self.assertEqual(positions, [])
            captured = [index for index, payload in enumerate(payloads)
                        if payload["generate_config"].get("gen_timeline")]
            self.assertEqual(captured, list(range(32, 40)))
            self.assertTrue(all(payloads[index]["generate_config"]["profile_step"] == 1 for index in captured))
            receipt = json.loads((directory / "profile-control.json").read_text())
            self.assertEqual(receipt["wave"], 5)
            self.assertEqual(receipt["owners"], 8)
            self.assertEqual(receipt["arming"], "request")
            self.assertEqual(receipt["addresses"], driver.owner_addresses("owner0:28100", 8))
            self.assertTrue(receipt["clean_timing"])
            (directory / "prefill_wr0_1.json").write_text(json.dumps({
                "traceEvents": [{"cat": "kernel", "name": "idle"}],
            }))
            with patch.dict("sys.modules", httpx=SimpleNamespace(AsyncClient=Client)), patch.object(driver, "load", load):
                with self.assertRaisesRegex(RuntimeError, "different model step"):
                    await profiler.load(config)


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class ReferenceProtocolTest(unittest.TestCase):
    def test_first_dp_admission_gate_is_bounded_and_preserves_later_waves(self):
        case = driver.case_config("prefill", 4096, "dp8-compact140")
        env = {driver.DP_PREQUEUE_ENV: "1500", driver.DP_FIRST_PREQUEUE_ENV: "5000"}
        self.assertEqual(driver.dp_prequeue_ms(case, env), 1500)
        self.assertEqual(driver.dp_prequeue_ms(case, env, first_wave=True), 5000)
        self.assertEqual(driver.dp_prequeue_ms(case, {driver.DP_PREQUEUE_ENV: "1500"}, first_wave=True), 1500)
        for invalid in ("1499", "10001", "invalid"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                driver.dp_prequeue_ms(case, dict(env, **{driver.DP_FIRST_PREQUEUE_ENV: invalid}), first_wave=True)
        other = driver.case_config("prefill", 4096, "cp8-yoco-off")
        self.assertEqual(driver.dp_prequeue_ms(other, env, first_wave=True), 0)

    def test_dp_seed_preserves_pages_and_does_not_grow_past_target_capacity(self):
        class Tokenizer:
            def encode(self, text):
                return SimpleNamespace(ids=text.split() + ["lookahead"])

        prime = driver.prompt().split() + ["lookahead"]
        steps = list(driver.dp_prefill_prime_steps(Tokenizer(), prime))
        previous = 0
        for _, length, reused in steps:
            self.assertEqual(reused, previous)
            self.assertEqual((length - 1) % 256, 0)
            self.assertLessEqual(length - reused, result.FRESH)
            previous = length - 1
        self.assertEqual(previous, result.PREFIX)
        self.assertEqual(steps[0][1], 3841)
        self.assertEqual(steps[-1][1] - steps[-1][2], 257)

    def test_sweep_and_headroom(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(driver.main(["all", "--protocol", "reference-128k", "--dry-run"]), 0)
        cases = json.loads(output.getvalue())
        self.assertEqual([row["point"] for row in cases[:5]], list(result.POINTS["prefill"]))
        self.assertEqual([row["point"] for row in cases[5:]], list(result.POINTS["decode"]))
        self.assertEqual([row["dspark_gamma"] for row in cases[:5]], [5] * 5)
        self.assertEqual([row["dspark_gamma"] for row in cases[5:]], [3] * 6)
        self.assertEqual(cases[-1]["global_requests"], 2048)
        self.assertGreaterEqual(cases[-1]["max_seq_len"], result.CONTEXT + result.OUTPUT + 4)
        self.assertEqual(len(driver.prompt(2047).split()), result.CONTEXT - 1)

    def test_cached_prefill_profiles_align_total_work(self):
        for profile, cp, dp, batch, keep in (
            ("cp8-yoco-off", 8, 1, 8, 0),
            ("dp8-compact140", 1, 8, 1, 140),
        ):
            with self.subTest(profile=profile):
                case = driver.case_config("prefill", 4096, profile)
                self.assertEqual(case["point_unit"], "fresh_tokens_per_ppu")
                self.assertEqual(case["global_requests"], 8)
                self.assertEqual(case["fresh_tokens_total"], 32768)
                self.assertEqual(case["batch"], batch)
                self.assertEqual(case["yoco_keep_tokens"], keep)
                self.assertEqual(case["prefill_cp"], cp)
                self.assertEqual(case["prefill_dp"], dp)
                self.assertEqual(case["dspark_gamma"], 0)
                env = driver.role_env({
                    "case": case, "role": "prefill", "checkpoint": "/fake",
                    "node": {"address": "localhost:18090"}, "output": "/unused",
                })
                self.assertEqual(env["DP_SIZE"], str(dp))
                self.assertEqual(env["REUSE_CACHE"], "1")
                self.assertEqual(env["SP_TYPE"], "")
                self.assertEqual(env["GEN_NUM_PER_CIRCLE"], "0")
                self.assertEqual(env["DSV41_SWA_BOUNDED_REPLAY"], str(int(bool(keep))))
                self.assertEqual(env.get("RTP_V41_PPU_YOCO_KEEP_TOKENS"), str(keep) if keep else None)

        self.assertEqual(
            driver.owner_addresses("localhost:18090", 8),
            [f"localhost:{port}" for port in range(18090, 18162, 9)],
        )

    def test_single_request_per_dp_owner_is_a_selectable_cli_point(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(driver.main([
                "prefill", "--protocol", "reference-128k", "--dry-run",
                "--prefill-profile", "dp8-compact140", "--points", "4096",
            ]), 0)
        cases = json.loads(output.getvalue())
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["batch"], 1)
        self.assertEqual(cases[0]["global_requests"], 8)
        self.assertEqual(cases[0]["fresh_tokens_total"], 32768)

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

    def test_dp_prefill_requires_overlapping_full_cache_hits_on_eight_owners(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for rank in range(result.WORLD):
                rows = [dict(
                    step=wave * 7 + rank + 1, start_ns=(wave * 2 + rank / 100) * 10**9,
                    end_ns=(wave * 2 + 1 + rank / 100) * 10**9,
                    input_min=result.CONTEXT, input_max=result.CONTEXT, batch=1, context_batch=1,
                    prefix_min=result.PREFIX, prefix_max=result.PREFIX, fresh_tokens=4096, emitted_tokens=1,
                ) for wave in range(15)]
                write_rows(directory / f"step-rank{rank}.jsonl", rows)
            metrics = result.prefill_result(directory, 4096, owners=8)
            self.assertAlmostEqual(metrics["step_ms"], 1070)
            self.assertAlmostEqual(metrics["throughput_per_ppu"], 4096 / 1.07)
            self.assertEqual(metrics["sample_steps_by_rank"][7][0], 43)

            path = directory / "step-rank7.jsonl"
            rows[-1]["prefix_max"] = 0
            write_rows(path, rows)
            with self.assertRaisesRegex(ValueError, "rank 7: .*wrong cache hit"):
                result.prefill_result(directory, 4096, owners=8)
            rows[-1]["prefix_max"] = result.PREFIX
            rows[-1]["start_ns"] = rows[-1]["end_ns"] - 1
            write_rows(path, rows)
            with self.assertRaisesRegex(ValueError, "wave 14 has no common interval"):
                result.prefill_result(directory, 4096, owners=8)

    def test_prefill_rejects_wrong_replay_path(self):
        row = dict(
            lengths=[4096], prefixes=[result.PREFIX], plan="ordinary_full_prefill", compacted=False,
            dense_fp4_layers=[2, 8, 14, 20], sparse_fp4_layers=[24, 28, 32, 36],
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for rank in range(result.WORLD):
                write_rows(directory / f"prefill-rank{rank}.jsonl", [row] * 15)
            result.qualify_prefill_optimizations(directory, 4096, yoco=False)
            with self.assertRaisesRegex(ValueError, "expected compact140"):
                result.qualify_prefill_optimizations(directory, 4096)

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
