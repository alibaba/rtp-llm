import json
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

from navi_sched.adapters.sglang.NaviSchedSGLangPlugin import _run_batch
from navi_sched.adapters.sglang.SGLangObserver import SGLangObserver, workload_snapshot
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def request(rid, prompt, prefix=0, generated=0, finished=False):
    req = object.__new__(Req)
    req.rid = rid
    req.origin_input_ids = array("q", range(prompt))
    req.output_ids = [0] * generated
    req.prefix_len = prefix
    req.finished_reason = "done" if finished else None
    req.sampling_params = SimpleNamespace(max_new_tokens=20)
    return req


class SGLangObserverTest(unittest.TestCase):
    def test_large_model_fingerprint_fits_native_resource_limit(self):
        model = {"model_type": "large", "config": "x" * 5000}
        observer = object.__new__(SGLangObserver)
        observer.scheduler = SimpleNamespace(
            server_args=SimpleNamespace(),
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(to_dict=lambda: model)
            ),
        )
        device = SimpleNamespace(name="test GPU", total_memory=1000, major=9, minor=0)
        with patch("torch.cuda.current_device", return_value=0), patch(
            "torch.cuda.get_device_properties", return_value=device
        ):
            first = observer._latency_fingerprint()
            self.assertRegex(first, r"^sglang:sha256:[0-9a-f]{64}$")
            self.assertEqual(observer._latency_fingerprint(), first)
            model["config"] += "y"
            self.assertNotEqual(observer._latency_fingerprint(), first)
            model["config"] = "x" * 5000
            self.assertEqual(observer._latency_fingerprint(), first)

    def test_latency_history_is_separated_by_model_and_execution_config(self):
        observer = object.__new__(SGLangObserver)
        args = SimpleNamespace(
            model_path="model-a", tp_size=1, disable_overlap_schedule=False
        )
        observer.scheduler = SimpleNamespace(
            server_args=args,
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(to_dict=lambda: {"hidden_size": 128})
            ),
        )
        device = SimpleNamespace(name="test GPU", total_memory=1000, major=9, minor=0)
        with patch("torch.cuda.current_device", return_value=0), patch(
            "torch.cuda.get_device_properties", return_value=device
        ):
            first = observer._latency_fingerprint()
            self.assertEqual(observer._latency_fingerprint(), first)
            args.model_path = "model-b"
            self.assertNotEqual(observer._latency_fingerprint(), first)
            args.model_path = "model-a"
            args.disable_overlap_schedule = True
            self.assertNotEqual(observer._latency_fingerprint(), first)
        values = json.loads(first)
        self.assertEqual(values["model_config"], {"hidden_size": 128})
        self.assertEqual(values["gpu_memory_bytes"], 1000)
        self.assertEqual(values["framework"], "sglang")

    def test_decode_timing_does_not_reuse_extend_token_count(self):
        observer = SimpleNamespace(
            enabled=True,
            prefill_started={},
            report=lambda *args, **kwargs: None,
            send=lambda value: None,
        )
        scheduler = SimpleNamespace(navi_sched=observer, enable_overlap=False)
        batch = SimpleNamespace(
            reqs=[SimpleNamespace(output_ids=[1])],
            extend_num_tokens=99,
            forward_mode=ForwardMode.DECODE,
        )
        with patch(
            "torch.cuda.Event",
            side_effect=lambda **kwargs: SimpleNamespace(record=lambda stream: None),
        ), patch("torch.cuda.current_stream", return_value=None):
            result = _run_batch(lambda *args: SimpleNamespace(), scheduler, batch)
        self.assertEqual(result.navi_sched_timing[2:], (0, 1))

    def test_verify_timing_keeps_decode_phase_without_reading_acceptance(self):
        for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
            with self.subTest(mode=mode):
                observer = SimpleNamespace(
                    enabled=True,
                    prefill_started={},
                    report=lambda *args, **kwargs: None,
                    send=lambda value: None,
                )
                scheduler = SimpleNamespace(navi_sched=observer, enable_overlap=False)
                batch = SimpleNamespace(
                    reqs=[SimpleNamespace(output_ids=[1])],
                    extend_num_tokens=99,
                    forward_mode=mode,
                )
                acceptance = object()
                native_result = SimpleNamespace(accept_lens=acceptance)

                def native_forward(*args):
                    batch.forward_mode = ForwardMode.TARGET_VERIFY
                    return native_result

                with patch(
                    "torch.cuda.Event",
                    side_effect=lambda **kwargs: SimpleNamespace(
                        record=lambda stream: None
                    ),
                ), patch("torch.cuda.current_stream", return_value=None):
                    result = _run_batch(native_forward, scheduler, batch)
                self.assertIs(result, native_result)
                self.assertIs(result.accept_lens, acceptance)
                self.assertEqual(result.navi_sched_timing[2:], (0, 1))

    def test_speculative_configuration_separates_latency_history(self):
        observer = object.__new__(SGLangObserver)
        args = SimpleNamespace(
            speculative_algorithm=None,
            speculative_dspark_block_size=5,
            speculative_draft_model_path="draft-a",
        )
        observer.scheduler = SimpleNamespace(
            server_args=args,
            model_config=SimpleNamespace(hf_config=SimpleNamespace(to_dict=dict)),
        )
        device = SimpleNamespace(name="test GPU", total_memory=1000, major=9, minor=0)
        with patch("torch.cuda.current_device", return_value=0), patch(
            "torch.cuda.get_device_properties", return_value=device
        ):
            ordinary = observer._latency_fingerprint()
            args.speculative_algorithm = "DSPARK"
            dspark = observer._latency_fingerprint()
            self.assertNotEqual(dspark, ordinary)
            self.assertEqual(
                json.loads(dspark)["speculative"],
                {
                    "speculative_algorithm": "DSPARK",
                    "speculative_dspark_block_size": 5,
                    "speculative_draft_model_path": "draft-a",
                },
            )
            args.speculative_dspark_block_size = 8
            self.assertNotEqual(observer._latency_fingerprint(), dspark)
            args.speculative_dspark_block_size = 5
            args.speculative_draft_model_path = "draft-b"
            self.assertNotEqual(observer._latency_fingerprint(), dspark)

    def test_buffer_loads_block_device_readiness(self):
        observer = object.__new__(SGLangObserver)
        pipeline = SimpleNamespace(ongoing_buffer_load_back={-1: object()})
        observer.scheduler = SimpleNamespace(
            tree_cache=SimpleNamespace(ongoing_load_back={}, buffer_pipeline=pipeline)
        )
        self.assertTrue(observer._has_pending_loads())
        pipeline.ongoing_buffer_load_back.clear()
        self.assertFalse(observer._has_pending_loads())

    def test_target_verification_remains_decode_workload(self):
        req = request("verify", 10, prefix=10, generated=4)
        scheduler = SimpleNamespace(
            running_batch=SimpleNamespace(reqs=[req]),
            waiting_queue=[],
            chunked_req=None,
            max_running_requests=8,
        )
        batch = SimpleNamespace(reqs=[req], forward_mode=ForwardMode.TARGET_VERIFY)
        values = workload_snapshot(scheduler, batch)
        self.assertEqual(values["remaining_prefill_request_count"], 0)
        self.assertEqual(values["remaining_prefill_token_count"], 0)
        self.assertEqual(values["running_decode_request_count"], 1)
        self.assertEqual(values["running_decode_generated_token_count"], 4)
        self.assertEqual(values["running_decode_remaining_token_budget"], 16)

    def test_prefill_decode_waiting_and_chunk_progress(self):
        prefill = request("prefill", 12, prefix=4)
        decode = request("decode", 10, generated=3)
        chunk = request("chunk", 24, prefix=16)
        waiting = request("waiting", 8, prefix=2)
        done = request("done", 100, generated=20, finished=True)
        scheduler = SimpleNamespace(
            running_batch=SimpleNamespace(reqs=[decode, done]),
            waiting_queue=[prefill, waiting],
            chunked_req=chunk,
            max_running_requests=8,
        )
        batch = SimpleNamespace(reqs=[prefill], forward_mode=ForwardMode.EXTEND)
        values = workload_snapshot(scheduler, batch)
        self.assertEqual(values["request_count"], 4)
        self.assertEqual(values["running_request_count"], 3)
        self.assertEqual(values["waiting_request_count"], 1)
        self.assertEqual(values["running_capacity_prefill_context_token_count"], 20)
        self.assertEqual(values["running_capacity_prefill_token_count"], 16)
        self.assertEqual(values["running_capacity_prefill_token_square_sum"], 128)
        self.assertEqual(values["waiting_capacity_prefill_token_count"], 6)
        self.assertEqual(values["remaining_prefill_token_count"], 22)
        self.assertEqual(values["context_remaining_prefill_token_product_sum"], 172)
        self.assertEqual(values["running_decode_request_count"], 1)
        self.assertEqual(values["running_decode_generated_token_count"], 3)
        self.assertEqual(values["running_decode_remaining_token_budget"], 17)
        self.assertEqual(values["total_prompt_token_count"], 54)

    def test_retracted_decode_requires_prefill_again(self):
        req = request("retracted", 10, prefix=4, generated=3)
        scheduler = SimpleNamespace(
            running_batch=SimpleNamespace(reqs=[]),
            waiting_queue=[req],
            chunked_req=None,
            max_running_requests=8,
        )
        values = workload_snapshot(scheduler)
        self.assertEqual(values["waiting_capacity_prefill_token_count"], 9)
        self.assertEqual(values["remaining_prefill_context_token_count"], 4)
        self.assertEqual(values["running_decode_request_count"], 0)


if __name__ == "__main__":
    unittest.main()
