"""Execute the native publisher and PyWrappedModel with real torch tensors.

Requires the built extension and CUDA; FP8 cases also require the real TorchSpec
quantizer. Only the external Mooncake store, legacy network writer and remote TP
peers are substituted. The Python model supplies tensors, not wrapper behavior.
No source-contract tests, extracted code, dependency skips or runtime fallbacks.
"""

import importlib
import importlib.util
import sys
import types
import unittest
from itertools import product
from pathlib import Path
from unittest.mock import Mock, patch


class MockStore:
    instances = []
    failure = None

    def __init__(self, config):
        self.config = config
        self.batches = []
        self.errors = []
        self.closed = False
        self.flushes = 0
        self.events = []
        self.instances.append(self)

    def setup(self, device):
        self.device = device

    def warmup_rdma(self):
        if self.failure == "init":
            raise RuntimeError("injected warmup failure")

    def take_async_errors(self):
        self.events.append("poll")
        errors, self.errors = self.errors, []
        return errors

    def put_batch(self, **payload):
        self.events.append("put")
        if self.failure == "admit":
            raise RuntimeError("injected admission failure")
        if self.failure == "duplicate":
            raise FileExistsError("duplicate request key")
        self.batches.append(payload)
        if self.failure == "async":
            self.errors.append(
                RuntimeError(f"batch {payload['batch_id']} completion failed")
            )
        return (
            [] if self.failure == "contract" else [{} for _ in payload["request_ids"]]
        )

    def flush(self):
        self.events.append("flush")
        self.flushes += 1
        if self.failure == "flush":
            raise RuntimeError("injected flush failure")

    def query_keys(self, keys):
        # This fake models an immediate receiver taking ownership and deleting
        # every source key, so successive fixture forwards may reserve again.
        return {key: False for key in keys}

    def close(self):
        self.events.append("close")
        self.closed = True


class MockConfig:
    hidden_dim = 2
    num_aux_layers = 2
    store_key_namespace = "GlmBase"

    @classmethod
    def from_env(cls):
        return cls()

    def make_store_key(self, request_id):
        return f"rtp_{self.store_key_namespace}_{request_id}"


class CaptureFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch = importlib.import_module("torch")
        cls.ext = importlib.import_module(
            "rtp_llm.cpp.models.test.libth_pywrapped_model_cache_store_integration_test"
        )

    def setUp(self):
        MockStore.instances = []
        MockStore.failure = None
        mooncake = types.ModuleType("torchspec.transfer.mooncake")
        mooncake.MooncakeConfig = MockConfig
        mooncake.EagleMooncakeStore = MockStore
        # Do not replace torchspec.utils.fp8: the successful FP8 path must run
        # the installed quantizer, not a second implementation of its algorithm.
        # The store is mocked; importing its real parent eagerly loads training
        # models and unrelated Torch registrations. Keep the real package path
        # so torchspec.utils.fp8 still executes the installed quantizer below.
        existing_torchspec = sys.modules.get("torchspec")
        torchspec_spec = (
            importlib.util.find_spec("torchspec")
            if existing_torchspec is None
            else getattr(existing_torchspec, "__spec__", None)
        )
        torchspec = types.ModuleType("torchspec")
        torchspec.__path__ = (
            list(getattr(existing_torchspec, "__path__", []))
            if existing_torchspec is not None
            else (
                list(torchspec_spec.submodule_search_locations)
                if torchspec_spec is not None
                else []
            )
        )
        transfer = types.ModuleType("torchspec.transfer")
        transfer.__path__ = []
        self.module_patch = patch.dict(
            sys.modules,
            {
                "torchspec": torchspec,
                "torchspec.transfer": transfer,
                mooncake.__name__: mooncake,
            },
        )
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.publishers = []
        self.addCleanup(self.close_publishers)
        self.addCleanup(self.ext.clear_comm_ops)

    def close_publishers(self):
        for publisher in self.publishers:
            publisher.close_without_gil()
        self.publishers.clear()

    def publisher(self, **kwargs):
        publisher = self.ext.CapturePublisher(**kwargs)
        self.publishers.append(publisher)
        return publisher

    def payload(self, device="cpu"):
        t = self.torch
        return [
            t.arange(30, dtype=t.float16, device=device).reshape(5, 6),
            t.arange(5, dtype=t.int32, device=device),
            t.tensor([2, 3], dtype=t.int32),
            t.tensor([100, 101], dtype=t.int64),
        ]

    def model(
        self,
        malformed=None,
        malformed_once=False,
        writer_factory=None,
        mtp=False,
        final_norm=False,
        capture_layer_ids=(0, 1),
        mtp_failure=None,
        mtp_hc_mult=1,
        capture_error=None,
        final_failure=None,
        capability_failure=None,
    ):
        t, ext = self.torch, self.ext

        class Model:
            supports_hidden_state_capture = True
            micro_batch_final_norm = True

            def __init__(self):
                self.flags = []
                self.cache_inputs = []
                self.attention_inputs = []
                self.token_ids = []
                self.writers = []
                self.mtp_hidden = None
                self.mtp_calls = []
                self.resources = None

            def initialize(self, resources):
                self.resources = resources
                return True

            def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
                return None

            def get_mtp_target_hidden_states(self, num_tokens):
                self.mtp_calls.append(num_tokens)
                if mtp_failure == "exception":
                    raise RuntimeError("injected MTP getter failure")
                if mtp_failure == "rows":
                    return self.mtp_hidden[:-1]
                if mtp_failure == "rank":
                    return self.mtp_hidden.flatten()
                if mtp_failure == "width":
                    return self.mtp_hidden[:, :-1].contiguous()
                if mtp_failure == "dtype":
                    return self.mtp_hidden.float()
                if mtp_failure == "device":
                    return self.mtp_hidden.cpu()
                if mtp_failure == "empty":
                    return self.mtp_hidden[:0]
                return self.mtp_hidden if mtp else None

            def forward(self, inputs, fmha=None):
                self.flags.append(inputs.capture_hidden_states)
                self.cache_inputs.append(inputs.attention_inputs.cache_store_inputs)
                self.attention_inputs.append(inputs.attention_inputs)
                self.token_ids.append(inputs.input_ids.clone())
                if writer_factory is not None:
                    writer = writer_factory(inputs.attention_inputs)
                    self.writers.append(writer)
                    if writer is not None:
                        writer.forward(None)
                hidden = (
                    inputs.input_ids.to(t.float16)
                    .unsqueeze(1)
                    .expand(-1, 2)
                    .contiguous()
                )
                self.mtp_hidden = (hidden + 100).repeat(1, mtp_hc_mult)
                if final_failure is not None:
                    raise final_failure
                last_hidden = hidden + 30
                if final_norm:
                    # Python owns final norm in both eager and micro-batch forwards.
                    # Preserve the model dtype required by the packed capture layout.
                    gamma = t.tensor([0.5, 1.5], device=hidden.device, dtype=t.float32)
                    last_hidden = t.nn.functional.rms_norm(
                        last_hidden.float(), (2,), weight=gamma, eps=1e-5
                    ).to(hidden.dtype)
                packed = None
                if inputs.capture_hidden_states:
                    packed = t.cat(
                        [hidden + (layer_id + 1) * 10 for layer_id in capture_layer_ids]
                        + [last_hidden],
                        -1,
                    )
                    if not malformed_once or len(self.flags) == 1:
                        if malformed == "width":
                            packed = packed[:, :-1]
                        elif malformed == "rows":
                            packed = packed[:-1]
                        elif malformed == "rank":
                            packed = packed.flatten()
                        elif malformed == "dtype":
                            packed = packed.float()
                        elif malformed == "device":
                            packed = packed.cpu()
                        elif malformed == "absent":
                            packed = None
                output = ext.PyModelOutputs(last_hidden, params_ptr=fmha)
                output.captured_hidden_states = (
                    packed if capture_error is None else None
                )
                output.capture_error = (
                    capture_error if inputs.capture_hidden_states else None
                )
                return output

            def forward_micro_batch(self, inputs):
                outputs = [self.forward(item) for item in inputs]
                if malformed == "count":
                    return outputs[:-1]
                if malformed == "fake_lane":
                    outputs[-1] = ext.PyModelOutputs(
                        t.empty(0, device=inputs[0].input_ids.device)
                    )
                return outputs

        if capability_failure == "missing":
            del Model.supports_hidden_state_capture
        elif capability_failure == "false":
            Model.supports_hidden_state_capture = False
        elif capability_failure == "raise":

            def capability(instance):
                raise ValueError(
                    "injected rank-local capture capability initialization failure"
                )

            Model.supports_hidden_state_capture = property(capability)
        return Model()

    def register_peers(self, allreduce, allgather=None):
        def unexpected(*args):
            self.fail("unexpected TP collective")

        awaiting_capability = True

        def capture_consensus(tensor, op, mode, dest):
            nonlocal awaiting_capability
            if awaiting_capability:
                self.assertEqual((op, mode, dest), (3, 0, None))
                self.assertEqual(tensor.tolist(), [1])
                awaiting_capability = False
                return tensor
            awaiting_capability = True
            return allreduce(tensor, op, mode, dest)

        self.ext.register_comm_ops(
            unexpected, capture_consensus, allgather or unexpected
        )

    def cp_peers(
        self,
        mtp=False,
        logits=False,
        lengths=(6,),
        tp_size=2,
        tp_rank=0,
        input_ids_on_cuda=False,
        capture_layer_ids=(0, 1),
        mtp_hc_mult=1,
        capture_valid=True,
        local_capture_valid=None,
    ):
        t = self.torch
        calls = []
        ids = t.arange(
            1,
            sum(lengths) + 1,
            dtype=t.int32,
            device="cuda" if input_ids_on_cuda else "cpu",
        )
        peers = []
        for rank in range(tp_size):
            split = self.ext.split_cp_inputs(ids, lengths, tp_size, rank)
            self.assertEqual(split["original_lengths"].tolist(), list(lengths))
            peer = self.model(
                mtp=mtp, capture_layer_ids=capture_layer_ids, mtp_hc_mult=mtp_hc_mult
            )
            output = peer.forward(
                types.SimpleNamespace(
                    input_ids=split["input_ids"],
                    capture_hidden_states=True,
                    attention_inputs=types.SimpleNamespace(cache_store_inputs=None),
                )
            )
            peers.append(
                (output.captured_hidden_states, peer.mtp_hidden, output.hidden_states)
            )

        def allreduce(tensor, op, mode, dest):
            self.assertGreater(tp_size, 1)
            self.assertEqual((op, mode, dest), (0, 0, None))  # SUM, TP
            self.assertEqual(calls, [])
            calls.append("validate")
            local_valid = (
                capture_valid if local_capture_valid is None else local_capture_valid
            )
            self.assertEqual(tensor.tolist(), [int(local_valid), 1, int(mtp)])
            return tensor.new_tensor(
                [tp_size * int(capture_valid), tp_size, tp_size * int(mtp)]
            )

        def allgather(receive, mode, send, inplace):
            self.assertGreater(tp_size, 1)
            self.assertEqual(mode, 0)
            self.assertFalse(inplace)
            self.assertEqual(calls[0], "validate")
            if send[0].shape[1] == (len(capture_layer_ids) + 1) * 2:
                self.assertTrue(capture_valid, "invalid capture must skip its gather")
                calls.append("capture_restore")
                payloads = [peer[0] for peer in peers]
            elif mtp and "mtp_restore" not in calls:
                self.assertEqual(send[0].shape[1], 2 * mtp_hc_mult)
                calls.append("mtp_restore")
                payloads = [peer[1] for peer in peers]
            elif "normal_restore" not in calls:
                calls.append("normal_restore")
                payloads = [peer[2] for peer in peers]
            else:
                self.assertTrue(logits)
                calls.append("logits")
                payloads = [send[0]] * tp_size
            t.testing.assert_close(send[0], payloads[tp_rank])
            payloads[tp_rank] = send[0]
            receive[0].copy_(t.cat(payloads, 0))

        self.register_peers(allreduce, allgather)
        return calls


class PublisherTest(CaptureFixture):
    def ready_publisher(self, **kwargs):
        publisher = self.publisher(**kwargs)
        publisher.flush()  # Explicit test-only barrier; never a serving call.
        return publisher

    def test_bf16_request_slicing_and_close_without_gil(self):
        publisher = self.ready_publisher()
        payload = self.payload()
        result = publisher.forward(*payload)
        self.assertTrue(result["accepted"])
        self.assertIsNone(publisher.take_error())
        batch = MockStore.instances[-1].batches[0]
        self.assertEqual(batch["request_ids"], ["rtp_GlmBase_100", "rtp_GlmBase_101"])
        for index, (start, end) in enumerate(((0, 2), (2, 5))):
            self.torch.testing.assert_close(
                batch["hidden_states"][index], payload[0][start:end, :4].bfloat16()
            )
            self.torch.testing.assert_close(
                batch["last_hidden_states"][index], payload[0][start:end, 4:].bfloat16()
            )
            self.assertEqual(batch["input_ids"][index].dtype, self.torch.int64)
            self.assertEqual(
                batch["input_ids"][index].tolist(), payload[1][start:end].tolist()
            )
        self.assertNotIn("hidden_states_scale", batch)
        self.assertEqual(batch["capture_metadata"]["capture_layer_ids"], [0, 1])
        publisher.close_without_gil()
        store = MockStore.instances[-1]
        self.assertTrue(store.closed)
        events = list(store.events)
        publisher.close_without_gil()
        self.assertEqual(store.events, events)

    def test_background_put_does_not_block_submission_and_close_releases_gil(self):
        from threading import Event

        publisher = self.ready_publisher()
        entered, release, finished = Event(), Event(), Event()
        original = MockStore.put_batch

        def blocked_put(store, **payload):
            entered.set()
            if not release.wait(timeout=10):
                raise RuntimeError("test PUT gate timed out")
            result = original(store, **payload)
            finished.set()
            return result

        with patch.object(MockStore, "put_batch", blocked_put):
            try:
                result = publisher.forward(*self.payload(), wait=False)
                self.assertTrue(result["accepted"])
                self.assertTrue(entered.wait(timeout=10))
                self.assertFalse(finished.is_set())
                self.assertIsNone(publisher.take_error())
            finally:
                release.set()
                publisher.close_without_gil()
        self.assertTrue(finished.is_set())
        self.assertTrue(MockStore.instances[-1].closed)

    def test_shutdown_flush_failure_still_closes_store(self):
        publisher = self.ready_publisher()
        MockStore.failure = "flush"
        publisher.close_without_gil()
        self.assertTrue(MockStore.instances[-1].closed)
        self.assertIsNone(publisher.take_error())

    def test_non_owner_does_not_initialize_store_or_reserve(self):
        publisher = self.ready_publisher(owner=False)
        self.assertFalse(publisher.reserve(100, 2))
        self.assertFalse(publisher.forward(*self.payload())["accepted"])
        self.assertEqual(MockStore.instances, [])
        self.assertIsNone(publisher.take_error())

    def test_reservation_rejects_invalid_and_duplicate_ids_and_cancel_is_idempotent(
        self,
    ):
        publisher = self.ready_publisher()
        for request, tokens in ((-1, 1), (100, 0), (100, -1), (100, 2**62)):
            self.assertFalse(publisher.reserve(request, tokens))
        self.assertTrue(publisher.reserve(100, 2))
        self.assertFalse(publisher.reserve(100, 2))
        publisher.cancel(100)
        publisher.cancel(100)
        self.assertTrue(publisher.reserve(100, 2))
        publisher.cancel(100)

    def test_metadata_and_layout_rejections_never_return_serving_deferred_errors(self):
        for index, change in (
            (0, lambda x: x.flatten()),
            (0, lambda x: x[:, :-1]),
            (0, lambda x: x[:-1]),
            (0, lambda x: x.float()),
            (1, lambda x: x.reshape(1, -1)),
            (2, lambda x: x.long()),
            (3, lambda x: x.int()),
            (2, lambda x: x.reshape(1, -1)),
            (3, lambda x: x.reshape(1, -1)),
        ):
            with self.subTest(index=index):
                publisher = self.ready_publisher()
                payload = self.payload()
                payload[index] = change(payload[index])
                self.assertFalse(publisher.forward(*payload)["accepted"])
                self.assertIsNone(publisher.take_error())
                self.assertEqual(MockStore.instances[-1].batches, [])
                publisher.close_without_gil()

    def test_store_failures_are_background_telemetry_not_serving_errors(self):
        for failure in ("admit", "duplicate", "flush", "async", "contract"):
            with self.subTest(failure=failure):
                publisher = self.ready_publisher()
                MockStore.failure = failure
                result = publisher.forward(*self.payload())
                self.assertTrue(result["accepted"])
                self.assertIsNone(publisher.take_error())
                self.assertGreaterEqual(result["failures"], 1)
                publisher.close_without_gil()
                MockStore.failure = None

    def test_initialization_failure_disables_capture_not_construction(self):
        MockStore.failure = "init"
        publisher = self.ready_publisher()
        self.assertFalse(publisher.enabled())
        self.assertFalse(publisher.reserve(100, 2))
        self.assertIsNone(publisher.take_error())
        publisher.close_without_gil()
        self.assertTrue(MockStore.instances[-1].closed)

    def test_noncontiguous_payload_and_single_token_boundary(self):
        publisher = self.ready_publisher()
        payload = self.payload()
        payload[0] = payload[0].t().contiguous().t()
        self.assertFalse(payload[0].is_contiguous())
        self.assertTrue(publisher.forward(*payload)["accepted"])
        single = [
            payload[0][:1],
            payload[1][:1],
            self.torch.tensor([1], dtype=self.torch.int32),
            self.torch.tensor([102], dtype=self.torch.int64),
        ]
        self.assertTrue(publisher.forward(*single)["accepted"])
        self.assertEqual(
            MockStore.instances[-1].batches[-1]["input_ids"][0].tolist(), [0]
        )

    def test_fp8_real_quantizer_scales_and_bf16_last_hidden(self):
        quantizer = importlib.import_module(
            "torchspec.utils.fp8"
        ).quantize_aux_hidden_states
        payload = self.payload(device="cuda")
        expected_hidden, expected_scale = quantizer(payload[0][:, :4].contiguous(), 2)
        publisher = self.ready_publisher(fp8=True)
        self.assertTrue(publisher.forward(*payload)["accepted"])
        self.assertIsNone(publisher.take_error())
        batch = MockStore.instances[-1].batches[0]
        self.assertEqual(batch["capture_metadata"]["capture_layer_ids"], [0, 1])
        self.assertEqual(batch["hidden_states"][0].dtype, self.torch.float8_e4m3fn)
        self.torch.testing.assert_close(
            self.torch.cat(batch["hidden_states"]).float(), expected_hidden.float()
        )
        self.torch.testing.assert_close(
            self.torch.cat(batch["hidden_states_scale"]), expected_scale
        )
        self.torch.testing.assert_close(
            self.torch.cat(batch["last_hidden_states"]), payload[0][:, 4:].bfloat16()
        )

    def test_invalid_or_failed_fp8_quantizer_never_creates_serving_error(self):
        module = importlib.import_module("torchspec.utils.fp8")
        for failure in (ValueError("quantizer unavailable"), "not tensors"):
            with self.subTest(failure=type(failure).__name__):
                patch_kwargs = (
                    {"side_effect": failure}
                    if isinstance(failure, Exception)
                    else {"return_value": failure}
                )
                with patch.object(module, "quantize_aux_hidden_states", **patch_kwargs):
                    publisher = self.ready_publisher(fp8=True)
                    result = publisher.forward(*self.payload(device="cuda"))
                    self.assertTrue(result["accepted"])
                    self.assertGreaterEqual(result["failures"], 1)
                    self.assertIsNone(publisher.take_error())
                    self.assertEqual(MockStore.instances[-1].batches, [])
                    publisher.close_without_gil()


class PyWrappedModelTest(CaptureFixture):
    def expected_hidden(self, lengths=(2, 3), mtp=False, hc_mult=1):
        t = self.torch
        ids = t.arange(1, sum(lengths) + 1, dtype=t.float16, device="cuda")
        return (ids + (100 if mtp else 30)).unsqueeze(1).expand(-1, 2 * hc_mult)

    def test_capture_and_skip_lm_head_are_independent(self):
        for capture, skip, micro_batch in product((False, True), repeat=3):
            with self.subTest(capture=capture, skip=skip, micro_batch=micro_batch):
                model = self.model()
                before = len(MockStore.instances)
                result = self.ext.run_model(
                    model, capture=capture, skip_lm_head=skip, micro_batch=micro_batch
                )
                self.assertIsNone(result["error"])
                self.assertIsNone(result["second_error"])
                self.assertEqual(result["logits"] is None, skip)
                self.assertEqual(model.flags, [capture] * (2 if micro_batch else 1))
                expected = self.expected_hidden()
                self.torch.testing.assert_close(result["hidden"], expected)
                if not skip:
                    self.torch.testing.assert_close(
                        result["logits"], expected[[1, 4]].float() * 2
                    )
                self.assertEqual(
                    [params is None for params in model.cache_inputs],
                    [skip] * len(model.flags),
                )
                stores = MockStore.instances[before:]
                self.assertTrue(all(store.closed for store in stores))
                self.assertEqual(
                    sum(len(store.batches) for store in stores), int(capture)
                )

    def test_cuda_request_lengths_and_ids_publish_without_disabling_normal_or_legacy_prefill(
        self,
    ):
        for micro_batch, skip in product((False, True), repeat=2):
            with self.subTest(micro_batch=micro_batch, skip=skip):
                before = len(MockStore.instances)
                model = self.model()
                result = self.ext.run_model(
                    model,
                    input_ids_on_cuda=True,
                    micro_batch=micro_batch,
                    skip_lm_head=skip,
                )
                self.assertEqual(model.flags, [True] * (2 if micro_batch else 1))
                self.assertIsNone(result["error"])
                self.torch.testing.assert_close(
                    result["hidden"], self.expected_hidden()
                )
                stores = MockStore.instances[before:]
                self.assertEqual(len(stores), 1)
                self.assertEqual(len(stores[0].batches), 1)
                batch = stores[0].batches[0]
                self.assertEqual(
                    [x.tolist() for x in batch["input_ids"]], [[1, 2], [3, 4, 5]]
                )
                if skip:
                    self.assertIsNone(result["logits"])
                else:
                    self.torch.testing.assert_close(
                        result["logits"], self.expected_hidden()[[1, 4]].float() * 2
                    )

    def test_custom_capture_limits_reach_publisher_and_only_gate_capture(self):
        defaults = [64, 1073741824, 131072, 8, 1000]
        cases = [("exact-budget", [2, 100, 5, 1, 1], True)]
        for index, value in enumerate((1, 99, 4, 0, 0)):
            limits = defaults.copy()
            limits[index] = value
            cases.append((f"limit-{index}", limits, False))
        for name, limits, accepted in cases:
            with self.subTest(name=name):
                before = len(MockStore.instances)
                model = self.model()
                result = self.ext.run_model(
                    model, capture_limits=limits, skip_lm_head=False
                )
                self.assertEqual(model.flags, [accepted])
                self.assertIsNone(result["error"])
                self.torch.testing.assert_close(
                    result["hidden"], self.expected_hidden()
                )
                self.torch.testing.assert_close(
                    result["logits"], self.expected_hidden()[[1, 4]].float() * 2
                )
                self.assertEqual(
                    sum(len(store.batches) for store in MockStore.instances[before:]),
                    int(accepted),
                )

    def test_unconfigured_warmup_fake_and_unsupported_capture_keep_normal_output(self):
        for micro_batch, mode in product(
            (False, True), ("unconfigured", "warmup", "fake", "unsupported")
        ):
            with self.subTest(micro_batch=micro_batch, mode=mode):
                model = self.model()
                kwargs = {mode: True} if mode in ("warmup", "fake") else {}
                if mode == "unsupported":
                    model.supports_hidden_state_capture = False
                result = self.ext.run_model(
                    model,
                    micro_batch=micro_batch,
                    configured=mode != "unconfigured",
                    **kwargs,
                )
                self.assertEqual(model.flags, [False] * (2 if micro_batch else 1))
                self.assertIsNone(result["error"])
                self.torch.testing.assert_close(
                    result["hidden"], self.expected_hidden()
                )

    def test_micro_batch_fake_empty_and_single_token_lanes(self):
        for lengths in ([2, 3], [1, 2, 3], [3], [1], [], [0]):
            with self.subTest(lengths=lengths):
                capture = sum(lengths) > 0
                model = self.model(
                    malformed="fake_lane" if len(lengths) == 1 and capture else None
                )
                result = self.ext.run_model(
                    model,
                    capture=capture,
                    configured=capture,
                    micro_batch=True,
                    lengths=lengths,
                )
                self.assertEqual(model.flags, [capture, capture])
                self.torch.testing.assert_close(
                    result["hidden"], self.expected_hidden(lengths)
                )
                self.assertIsNone(result["error"])
                if capture:
                    batch = MockStore.instances[-1].batches[0]
                    self.assertEqual(
                        [x.shape[0] for x in batch["hidden_states"]], lengths
                    )
                    self.assertEqual(
                        self.torch.cat(batch["input_ids"]).tolist(),
                        list(range(1, sum(lengths) + 1)),
                    )

    def test_capture_and_micro_batch_do_not_repeat_final_norm(self):
        t = self.torch
        gamma = t.tensor([0.5, 1.5], dtype=t.float32, device="cuda")
        raw = self.expected_hidden().float()
        expected = (
            raw * t.rsqrt(raw.square().mean(-1, keepdim=True) + 1e-5) * gamma
        ).half()
        for capture, micro_batch, skip in product((False, True), repeat=3):
            with self.subTest(capture=capture, micro_batch=micro_batch, skip=skip):
                kwargs = dict(
                    capture=capture, micro_batch=micro_batch, skip_lm_head=skip
                )
                unnormalized = self.ext.run_model(
                    self.model(), final_norm=True, **kwargs
                )
                t.testing.assert_close(unnormalized["hidden"], raw.half())
                baseline = self.ext.run_model(self.model(final_norm=True), **kwargs)
                result = self.ext.run_model(
                    self.model(final_norm=True), final_norm=True, **kwargs
                )
                t.testing.assert_close(result["hidden"], expected)
                t.testing.assert_close(result["hidden"], baseline["hidden"])
                if not skip:
                    t.testing.assert_close(
                        result["logits"],
                        expected[[1, 4]].sum(-1, keepdim=True).expand(-1, 2).float(),
                    )
                if capture:
                    t.testing.assert_close(
                        t.cat(MockStore.instances[-1].batches[0]["last_hidden_states"]),
                        expected.bfloat16(),
                    )

    def test_recoverable_capture_failures_preserve_canonical_hidden_and_logits(self):
        for micro_batch, skip, fault in product(
            (False, True),
            (False, True),
            ("width", "rows", "rank", "dtype", "device", "absent", "hook"),
        ):
            with self.subTest(micro_batch=micro_batch, skip=skip, fault=fault):
                before = len(MockStore.instances)
                model = self.model(
                    malformed=None if fault == "hook" else fault,
                    capture_error=(
                        "injected recoverable hook failure" if fault == "hook" else None
                    ),
                )
                result = self.ext.run_model(
                    model, micro_batch=micro_batch, skip_lm_head=skip
                )
                self.assertIsNone(result["error"])
                self.assertIsNone(result["second_error"])
                self.torch.testing.assert_close(
                    result["hidden"], self.expected_hidden()
                )
                if not skip:
                    self.torch.testing.assert_close(
                        result["logits"], self.expected_hidden()[[1, 4]].float() * 2
                    )
                self.assertTrue(
                    all(not store.batches for store in MockStore.instances[before:])
                )

    def test_capture_failure_does_not_poison_later_capture_on_same_model(self):
        result = self.ext.run_model(
            self.model(malformed="width", malformed_once=True),
            forward_captures=[True, False, True],
        )
        for forward in result["forwards"]:
            self.assertIsNone(forward["error"])
            self.assertEqual(forward["broken_rejections"], 0)
            self.torch.testing.assert_close(forward["hidden"], self.expected_hidden())
        self.assertEqual(len(MockStore.instances[-1].batches), 1)

    def test_background_store_failure_does_not_poison_next_serving_or_capture(self):
        original = MockStore.put_batch
        attempts = []

        def fail_first(store, **payload):
            attempts.append(payload["batch_id"])
            if len(attempts) == 1:
                raise RuntimeError("injected operational admission failure")
            return original(store, **payload)

        with patch.object(MockStore, "put_batch", fail_first):
            result = self.ext.run_model(self.model(), forward_captures=[True, True])
        for forward in result["forwards"]:
            self.assertIsNone(forward["error"])
            self.torch.testing.assert_close(forward["hidden"], self.expected_hidden())
        self.assertEqual(len(attempts), 2)
        self.assertEqual(len(MockStore.instances[-1].batches), 1)

    def test_model_initialization_failure_remains_a_serving_failure(self):
        model = self.model()
        model.initialize = lambda resources: False
        with self.assertRaisesRegex(RuntimeError, "initialization failed"):
            self.ext.run_model(model)
        self.assertEqual(MockStore.instances, [])

    def test_missing_canonical_output_and_real_final_failure_are_serving_failures(self):
        for micro_batch in (False, True):
            with self.subTest(micro_batch=micro_batch):
                with self.assertRaisesRegex(ValueError, "real final failed"):
                    self.ext.run_model(
                        self.model(final_failure=ValueError("real final failed")),
                        micro_batch=micro_batch,
                    )
        with self.assertRaisesRegex(RuntimeError, "py_model_outputs.size"):
            self.ext.run_model(self.model(malformed="count"), micro_batch=True)

    def test_tp_capture_consensus_precedes_publication_and_preserves_normal_output(
        self,
    ):
        for micro_batch, owner, local_valid, peers_valid in product(
            (False, True), (False, True), (False, True), (False, True)
        ):
            with self.subTest(
                micro_batch=micro_batch,
                owner=owner,
                local=local_valid,
                peers=peers_valid,
            ):
                calls = []
                before = len(MockStore.instances)

                def allreduce(tensor, op, mode, dest):
                    self.assertEqual((op, mode, dest), (3, 0, None))
                    calls.append(tensor.tolist())
                    self.assertTrue(
                        all(not store.batches for store in MockStore.instances[before:])
                    )
                    flags = [int(local_valid and peers_valid)]
                    return tensor.new_tensor([1] + flags if micro_batch else flags)

                self.register_peers(allreduce)
                result = self.ext.run_model(
                    self.model(malformed=None if local_valid else "width"),
                    tp_size=2,
                    tp_rank=0 if owner else 1,
                    micro_batch=micro_batch,
                )
                expected_status = (
                    [1, int(local_valid)] if micro_batch else [int(local_valid)]
                )
                self.assertEqual(calls, [expected_status])
                self.assertIsNone(result["error"])
                self.torch.testing.assert_close(
                    result["hidden"], self.expected_hidden()
                )
                self.assertEqual(
                    sum(len(s.batches) for s in MockStore.instances[before:]),
                    int(owner and local_valid and peers_valid),
                )

    def assert_cp_batch(
        self, lengths, tp_size, tp_rank, input_ids_on_cuda, mtp, hc_mult=1
    ):
        t = self.torch
        layer_ids = (1, 0)
        before = len(MockStore.instances)
        calls = self.cp_peers(
            mtp=mtp,
            lengths=lengths,
            tp_size=tp_size,
            tp_rank=tp_rank,
            input_ids_on_cuda=input_ids_on_cuda,
            capture_layer_ids=layer_ids,
            mtp_hc_mult=hc_mult,
        )
        model = self.model(mtp=mtp, capture_layer_ids=layer_ids, mtp_hc_mult=hc_mult)
        if not mtp:
            del type(model).get_mtp_target_hidden_states
        result = self.ext.run_model(
            model,
            cp=True,
            tp_size=tp_size,
            tp_rank=tp_rank,
            lengths=lengths,
            input_ids_on_cuda=input_ids_on_cuda,
            capture_layer_ids=layer_ids,
            hc_mult=hc_mult,
        )
        expected_calls = (
            ["validate", "capture_restore"]
            + (["mtp_restore"] if mtp else [])
            + ["normal_restore"]
        )
        self.assertEqual(calls, expected_calls if tp_size > 1 else [])
        self.assertIsNone(result["error"])
        t.testing.assert_close(
            result["hidden"], self.expected_hidden(lengths, mtp, hc_mult if mtp else 1)
        )
        self.assertEqual(model.mtp_calls, [model.token_ids[0].numel()] if mtp else [])
        self.assertEqual(
            model.attention_inputs[
                0
            ].context_parallel_info.prefill_actual_input_lengths_cpu.tolist(),
            lengths,
        )
        stores = MockStore.instances[before:]
        self.assertEqual(len(stores), int(tp_rank == 0))
        if tp_rank != 0:
            return
        store = stores[0]
        self.assertTrue(store.closed)
        self.assertEqual(len(store.batches), 1)
        batch = store.batches[0]
        self.assertEqual(
            batch["capture_metadata"]["capture_layer_ids"], list(layer_ids)
        )
        self.assertEqual(
            batch["request_ids"],
            [f"rtp_GlmBase_{100 + i}" for i in range(len(lengths))],
        )
        ids = t.arange(1, sum(lengths) + 1, dtype=t.int64, device="cuda")
        base = ids.half().unsqueeze(1).expand(-1, 2)
        offset = 0
        for index, length in enumerate(lengths):
            request_hidden = base[offset : offset + length]
            t.testing.assert_close(
                batch["input_ids"][index], ids[offset : offset + length]
            )
            t.testing.assert_close(
                batch["hidden_states"][index],
                t.cat(
                    [request_hidden + (layer + 1) * 10 for layer in layer_ids], -1
                ).bfloat16(),
            )
            t.testing.assert_close(
                batch["last_hidden_states"][index], (request_hidden + 30).bfloat16()
            )
            offset += length

    def test_cp_multi_request_padding_order_all_ranks_and_expanded_mtp(self):
        for lengths, tp_size in (
            ([6, 9], 2),
            ([10, 20, 7], 4),
            ([1, 3], 4),
            ([8, 16], 4),
            ([1, 3], 1),
        ):
            for rank, cuda_ids, mtp in product(
                range(tp_size), (False, True), (False, True)
            ):
                with self.subTest(
                    lengths=lengths, size=tp_size, rank=rank, cuda_ids=cuda_ids, mtp=mtp
                ):
                    self.assert_cp_batch(
                        lengths, tp_size, rank, cuda_ids, mtp, hc_mult=4 if mtp else 1
                    )

    def test_cp_capture_rejection_skips_only_capture_gather_not_normal_or_mtp(self):
        for lengths, rank, mtp, local_fault, skip in product(
            ([6, 9], [1, 3]), range(2), (False, True), (False, True), (False, True)
        ):
            with self.subTest(
                lengths=lengths, rank=rank, mtp=mtp, local_fault=local_fault, skip=skip
            ):
                before = len(MockStore.instances)
                calls = self.cp_peers(
                    mtp=mtp,
                    lengths=lengths,
                    tp_rank=rank,
                    logits=not skip,
                    capture_valid=False,
                    local_capture_valid=not local_fault,
                )
                result = self.ext.run_model(
                    self.model(mtp=mtp, malformed="width" if local_fault else None),
                    cp=True,
                    tp_size=2,
                    tp_rank=rank,
                    lengths=lengths,
                    skip_lm_head=skip,
                )
                expected_calls = (
                    ["validate"]
                    + (["mtp_restore"] if mtp else [])
                    + ["normal_restore"]
                    + ([] if skip else ["logits"])
                )
                self.assertEqual(calls, expected_calls)
                self.assertIsNone(result["error"])
                self.torch.testing.assert_close(
                    result["hidden"], self.expected_hidden(lengths, mtp)
                )
                if not skip:
                    indexes = self.torch.tensor(lengths, device="cuda").cumsum(0) - 1
                    selected = self.expected_hidden(lengths)[indexes].float()
                    self.torch.testing.assert_close(
                        result["logits"], (selected * 2).repeat(1, 2)
                    )
                self.assertTrue(
                    all(not s.batches for s in MockStore.instances[before:])
                )

    def test_cp_mtp_failures_are_serving_failures_on_every_rank_even_with_bad_capture(
        self,
    ):
        for rank, fault, bad_capture in product(
            range(2),
            (
                "exception",
                "rows",
                "rank",
                "width",
                "dtype",
                "device",
                "absent",
                "empty",
                "no_getter",
            ),
            (False, True),
        ):
            with self.subTest(rank=rank, fault=fault, bad_capture=bad_capture):
                absent = fault in ("absent", "empty", "no_getter")
                local_status = [
                    int(not bad_capture),
                    1 if absent or rank == 0 else 0,
                    int(not absent or rank == 0),
                ]
                if fault == "exception" and rank == 1:
                    local_status[2] = 0

                def allreduce(tensor, op, mode, dest):
                    self.assertEqual((op, mode, dest), (0, 0, None))
                    self.assertEqual(tensor.tolist(), local_status)
                    return tensor.new_tensor(
                        [
                            0 if bad_capture else 2,
                            2 if absent else 1,
                            1 if absent or fault == "exception" else 2,
                        ]
                    )

                self.register_peers(allreduce)
                model = self.model(
                    mtp=not (rank == 1 and absent),
                    mtp_failure=fault if rank == 1 else None,
                    malformed="width" if bad_capture else None,
                )
                if rank == 1 and fault == "no_getter":
                    del type(model).get_mtp_target_hidden_states
                with self.assertRaisesRegex(RuntimeError, "CP MTP"):
                    self.ext.run_model(
                        model, cp=True, tp_size=2, tp_rank=rank, lengths=[6, 9]
                    )

    def test_tp_missing_canonical_micro_batch_output_fails_every_rank_before_publication(
        self,
    ):
        for local_count_valid in (False, True):
            with self.subTest(local_count_valid=local_count_valid):
                calls = []

                def allreduce(tensor, op, mode, dest):
                    self.assertEqual((op, mode, dest), (3, 0, None))
                    self.assertEqual(
                        tensor.tolist(), [1, 1] if local_count_valid else [0, 0]
                    )
                    calls.append("validate")
                    return tensor.new_tensor([0, 0])

                self.register_peers(allreduce)
                with self.assertRaisesRegex(RuntimeError, "py_model_outputs.size"):
                    self.ext.run_model(
                        self.model(malformed=None if local_count_valid else "count"),
                        tp_size=2,
                        micro_batch=True,
                    )
                self.assertEqual(calls, ["validate"])

    def test_ffn_capture_is_refused_before_normal_micro_batch_protocol(self):
        for micro_batch in (False, True):
            with self.subTest(micro_batch=micro_batch):
                before = len(MockStore.instances)
                model = self.model()
                result = self.ext.run_model(
                    model, ffn_disaggregate=True, micro_batch=micro_batch
                )
                self.assertEqual(model.flags, [False] * (2 if micro_batch else 1))
                self.assertIsNone(result["error"])
                self.torch.testing.assert_close(
                    result["hidden"], self.expected_hidden()
                )
                self.assertEqual(len(MockStore.instances), before)

    def test_capture_configuration_and_store_initialization_never_block_serving(self):
        for layer_ids in ([0, 0], [-1, 1], [0, 2]):
            result = self.ext.run_model(self.model(), capture_layer_ids=layer_ids)
            self.assertIsNone(result["error"])
            self.torch.testing.assert_close(result["hidden"], self.expected_hidden())
        MockStore.failure = "init"
        result = self.ext.run_model(self.model())
        self.assertIsNone(result["error"])
        self.torch.testing.assert_close(result["hidden"], self.expected_hidden())
        self.assertTrue(all(store.closed for store in MockStore.instances))

    def test_capability_initialization_errors_isolate_capture_but_not_gpu_failures(
        self,
    ):
        gpu_error = getattr(self.torch, "AcceleratorError", self.torch.OutOfMemoryError)
        for error in (
            ValueError("capture capability unavailable"),
            gpu_error("device execution failed"),
        ):
            with self.subTest(error=type(error).__name__):
                model = self.model()

                def capability(instance):
                    raise error

                type(model).supports_hidden_state_capture = property(capability)
                if isinstance(error, gpu_error):
                    with self.assertRaises(type(error)):
                        self.ext.run_model(model)
                else:
                    result = self.ext.run_model(model)
                    self.assertEqual(model.flags, [False])
                    self.assertIsNone(result["error"])
                    self.torch.testing.assert_close(
                        result["hidden"], self.expected_hidden()
                    )

    def test_micro_batch_requires_final_norm_contract_and_python_entry_point(self):
        model = self.model()
        model.micro_batch_final_norm = False
        with self.assertRaisesRegex(RuntimeError, "micro_batch_final_norm"):
            self.ext.run_model(model, micro_batch=True)
        model = self.model()
        del type(model).micro_batch_final_norm
        with self.assertRaisesRegex(RuntimeError, "micro_batch_final_norm"):
            self.ext.run_model(model, micro_batch=True)
        model = self.model()
        del type(model).forward_micro_batch
        with self.assertRaisesRegex(
            RuntimeError, "requires Python model method forward_micro_batch"
        ):
            self.ext.run_model(model, micro_batch=True)

    def test_micro_batch_final_norm_capability_is_cached_for_repeated_forwards(self):
        reads = []
        model = self.model()
        type(model).micro_batch_final_norm = property(
            lambda _instance: reads.append(True) or True
        )
        result = self.ext.run_model(
            model,
            micro_batch=True,
            forward_captures=[False, False, False],
        )
        # hasattr() and the cached attr() lookup both happen during initialization;
        # no lookup is made by any of the three forwards.
        self.assertEqual(len(reads), 2)
        self.assertEqual(model.flags, [False] * 6)
        self.assertIsNone(result["error"])

    def test_py_model_outputs_preserves_constructor_and_attention_ownership(self):
        import gc
        import weakref

        class Params:
            pass

        hidden = self.torch.ones((2, 2))
        packed = self.torch.ones((2, 6))
        for kwargs in (
            {},
            {"captured_hidden_states": packed},
            {"capture_error": "aux failed"},
        ):
            params = Params()
            ref = weakref.ref(params)
            output = self.ext.PyModelOutputs(
                hidden_states=hidden, params_ptr=params, **kwargs
            )
            self.assertIs(output.py_attn_params, params)
            self.assertIsNone(output.params_ptr)
            self.assertEqual(output.capture_error, kwargs.get("capture_error"))
            del params
            gc.collect()
            self.assertIsNotNone(ref())
            self.torch.testing.assert_close(output.hidden_states, hidden)
            del output
            gc.collect()
            self.assertIsNone(ref())
        for args in ((), (hidden,), (hidden, None), (hidden, self.ext.ParamsBase())):
            output = self.ext.PyModelOutputs(*args)
            self.assertIsNone(output.captured_hidden_states)
            self.assertIsNone(output.capture_error)
        self.assertIsInstance(output.params_ptr, self.ext.ParamsBase)

    def test_fp8_model_publication_uses_real_quantizer(self):
        result = self.ext.run_model(self.model(), fp8=True, micro_batch=True)
        self.assertIsNone(result["error"])
        batch = MockStore.instances[0].batches[0]
        self.assertEqual(
            [x.dtype for x in batch["hidden_states"]], [self.torch.float8_e4m3fn] * 2
        )
        self.assertEqual(
            [tuple(x.shape) for x in batch["hidden_states_scale"]], [(2, 2), (3, 2)]
        )
        self.torch.testing.assert_close(result["hidden"], self.expected_hidden())


class OldCacheStoreWrapperTest(CaptureFixture):
    """Load the complete legacy module; feed it native PyWrappedModel inputs."""

    def setUp(self):
        super().setUp()
        ops = types.ModuleType("rtp_llm.ops")
        ops.__path__ = []
        compute = types.ModuleType("rtp_llm.ops.compute_ops")
        for name in (
            "KVCache",
            "LayerKVCache",
            "PyAttentionInputs",
            "PyCacheStoreInputs",
        ):
            setattr(compute, name, getattr(self.ext, name))

        def write_cache_store(lengths, prefix, blocks, params, cache):
            return None

        self.write = Mock(wraps=write_cache_store)
        compute.write_cache_store = self.write
        ops.compute_ops = compute
        path = (
            Path(__file__).resolve().parents[4]
            / "rtp_llm/models_py/modules/base/common/kvcache_store.py"
        )
        spec = importlib.util.spec_from_file_location("legacy_kvcache_store_test", path)
        module = importlib.util.module_from_spec(spec)
        # Isolate package bootstrap, which otherwise loads a second copy of the
        # pybind types. Execute the whole module, including its real imports.
        with patch.dict(
            sys.modules, {"rtp_llm.ops": ops, "rtp_llm.ops.compute_ops": compute}
        ):
            spec.loader.exec_module(module)
        self.create = module.create_write_cache_store_impl

    def test_capture_skip_has_no_writer(self):
        model = self.model(writer_factory=self.create)
        self.ext.run_model(model)
        self.assertEqual(model.writers, [None])
        self.write.assert_not_called()

    def test_decode_has_no_writer(self):
        model = self.model(writer_factory=self.create)
        self.ext.run_model(model, capture=False, skip_lm_head=False, decode=True)
        self.assertEqual(model.writers, [None])
        self.write.assert_not_called()

    def test_old_five_argument_call_uses_native_host_metadata(self):
        model = self.model(writer_factory=self.create)
        self.ext.run_model(model, capture=False, skip_lm_head=False)
        self.write.assert_called_once()
        lengths, prefix, blocks, params, cache = self.write.call_args.args
        self.assertEqual(lengths.tolist(), [2, 3])
        self.assertEqual(prefix.tolist(), [0, 0])
        self.assertTrue(lengths.is_pinned())
        self.assertTrue(prefix.is_pinned())
        self.assertEqual(lengths.data_ptr(), params.input_lengths_host.data_ptr())
        self.assertEqual(prefix.data_ptr(), params.prefix_lengths_host.data_ptr())
        self.assertEqual(
            blocks.data_ptr(),
            model.attention_inputs[0].kv_cache_block_id_host.data_ptr(),
        )
        self.assertIsNone(cache)

    def test_cp_uses_original_lengths_not_local_split(self):
        calls = self.cp_peers(logits=True)
        model = self.model(writer_factory=self.create)
        result = self.ext.run_model(
            model, cp=True, tp_size=2, lengths=[6], skip_lm_head=False
        )
        self.assertIsNone(result["error"])
        self.assertEqual(
            calls, ["validate", "capture_restore", "normal_restore", "logits"]
        )
        self.write.assert_called_once()
        lengths, prefix, blocks, params, cache = self.write.call_args.args
        self.assertEqual(lengths.tolist(), [6])
        self.assertEqual(params.input_lengths_host.tolist(), [4])
        self.assertEqual(model.attention_inputs[0].input_lengths.tolist(), [4])


if __name__ == "__main__":
    unittest.main()
