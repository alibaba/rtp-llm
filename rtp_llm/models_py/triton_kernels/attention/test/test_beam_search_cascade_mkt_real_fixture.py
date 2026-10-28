"""Replay captured model tensors through the actual MI308 off/on attention.

MKT_CASCADE_FIXTURE_DIR must name a successful manual capture output directory.
Private prompt/token IDs/responses are neither read nor reported here. The saved
off output is checked bitwise with the same eager parameters before applying
the existing fixed FP32/FP64 accuracy gates to both paths.

Original eager cases retain Q strides and the whole shared KV backing allocation,
including each layer's storage offset. Graph replay uses a fixed anchor view and
copies actual layer tensors into it. Equivalent page/row permutations are labelled
derived fixtures; they are not new model layers or real scheduler batches.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import torch

from rtp_llm.models_py.triton_kernels.attention.test.test_beam_search_cascade_mkt import (
    AB_ATOL,
    AB_RELATIVE_L2_LIMIT,
    AB_RTOL,
    ATOL,
    DIM,
    HEADS,
    KV_HEADS,
    MAX_SEQ_LEN,
    PAGE,
    PRODUCTION_TAIL_KERNEL,
    REFERENCE_ATOL,
    REFERENCE_RTOL,
    RELATIVE_L2_LIMIT,
    RTOL,
    AttentionAB,
    MktCascadeTestCase,
    assert_accuracy,
    capture,
    cascade_mod,
    emit,
    error_metrics,
    finite_summary,
)
from rtp_llm.models_py.triton_kernels.attention.test.test_beam_search_cascade_mkt import (
    loaded_native_modules as _loaded_native_modules,
)
from rtp_llm.models_py.triton_kernels.attention.test.test_beam_search_cascade_mkt import (
    reference,
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_hash(tensor):
    payload = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return getattr(sys.modules.get(package), "__version__", "unavailable")


def _span_elements(metadata):
    shape, stride = metadata["shape"], metadata["stride"]
    _require(len(shape) == len(stride), "tensor shape/stride rank mismatch")
    _require(all(size > 0 for size in shape), "empty captured tensor")
    _require(all(step > 0 for step in stride), "unsupported captured tensor stride")
    return 1 + sum((size - 1) * step for size, step in zip(shape, stride))


def _check_layout(metadata):
    span = _span_elements(metadata)
    _require(metadata["dtype"] == "torch.bfloat16", "expected BF16 storage")
    _require(metadata["element_size"] == 2, "unexpected element size")
    _require(metadata["storage_offset"] >= 0, "negative storage offset")
    _require(span * 2 == metadata["view_span_bytes"], "invalid captured view span")
    _require(
        2 * (metadata["storage_offset"] + span) <= metadata["storage_nbytes"],
        "captured tensor extends beyond backing storage",
    )


def _unpack_pages(pages):
    """Independent inverse of the recorded BF16 vectorized byte layout."""
    count = pages.shape[0]
    key = pages[:, 0].contiguous().view(count, KV_HEADS, DIM // 8, PAGE, 8)
    key = key.permute(0, 1, 3, 2, 4).reshape(count, KV_HEADS, PAGE, DIM)
    value = pages[:, 1].contiguous().view(count, KV_HEADS, PAGE // 8, DIM, 8)
    value = value.permute(0, 1, 2, 4, 3).reshape(count, KV_HEADS, PAGE, DIM)
    return key, value


@dataclass
class CapturedFixture:
    manifest: dict
    tensors: dict
    manifest_sha256: str
    key: torch.Tensor
    value: torch.Tensor

    @property
    def name(self):
        m = self.manifest
        return f"request{m['request_ordinal']:04d}_layer{m['layer_idx']:03d}"

    def graph_key(self):
        m = self.manifest
        # Different original KV storage offsets are intentional: graphs retain
        # their anchor pointer and stage each source layer's live pages there.
        return (
            tuple(m["query_metadata"]["shape"]),
            tuple(m["query_metadata"]["stride"]),
            m["query_metadata"]["storage_offset"],
            tuple(m["paged_kv_metadata"]["shape"]),
            tuple(m["paged_kv_metadata"]["stride"]),
            tuple(self.tensors["block_table"].shape),
            m["max_seq_len"],
            m["configured_max_seq_len"],
        )


def load_fixtures(root):
    """CPU-only hash, schema, live-page mapping and finite-value validation."""
    summary = json.loads((root / "capture_summary.json").read_text())
    provenance_path = root / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance_hash = _sha_file(provenance_path)
    _require(summary["capture_status"] == "PASS", "capture was not successful")
    _require(
        summary["provenance_sha256"] == provenance_hash, "provenance hash mismatch"
    )
    _require(provenance["capture_mode"] == "eager", "only eager captures supported")
    manifests = sorted(root.glob("request_*/layer_*/manifest.json"))
    _require(len(manifests) == summary["fixture_count"] > 0, "incomplete fixture set")
    expected = {
        (request["ordinal"], layer): request["sha256"]
        for request in provenance["selected_requests"]
        for layer in provenance["selected_layers"]
    }
    fixtures, observed = [], set()
    for path in manifests:
        m = json.loads(path.read_text())
        identity = (m["request_ordinal"], m["layer_idx"])
        _require(
            identity in expected and identity not in observed,
            "invalid fixture identity",
        )
        observed.add(identity)
        _require(m["schema_version"] == 1, "unsupported capture schema")
        _require(m["request_sha256"] == expected[identity], "request hash mismatch")
        _require(
            m["provenance_sha256"] == provenance_hash, "fixture provenance mismatch"
        )
        _require(m["fixture_kind"] == "real_model_beam_capture", "not a model capture")
        _require(
            m["capture_mode"] == "eager" and not m["linear_v"],
            "unsupported capture mode",
        )
        _require(
            (m["head_num"], m["kv_head_num"], m["head_dim"], m["tokens_per_block"])
            == (HEADS, KV_HEADS, DIM, PAGE),
            "unsupported captured geometry",
        )
        _require(
            m["configured_max_seq_len"] == MAX_SEQ_LEN,
            "different model sequence capacity",
        )
        for name in ("query_metadata", "kv_base_metadata", "paged_kv_metadata"):
            _check_layout(m[name])
        _require(
            m["kv_base_metadata"] == m["paged_kv_metadata"],
            "KV base/view adaptation required",
        )
        _require(
            m["request_boundaries"] == [0, m["rows"]],
            "capture is not one sequential request",
        )
        tensor_file = path.parent / "tensors.pt"
        _require(
            _sha_file(tensor_file) == m["tensor_file_sha256"],
            "tensor file hash mismatch",
        )
        t = torch.load(tensor_file, map_location="cpu", weights_only=True)
        _require(set(t) == set(m["tensors"]), "saved tensor keys mismatch")
        for name, tensor in t.items():
            metadata = m["tensors"][name]
            _require(isinstance(tensor, torch.Tensor), "non-tensor payload")
            _require(
                list(tensor.shape) == metadata["shape"], "saved tensor shape mismatch"
            )
            _require(
                str(tensor.dtype) == metadata["dtype"], "saved tensor dtype mismatch"
            )
            _require(
                _tensor_hash(tensor) == metadata["sha256"], "saved tensor hash mismatch"
            )
        rows = m["rows"]
        _require(
            list(t["q"].shape) == m["query_metadata"]["shape"], "original Q shape lost"
        )
        _require(
            tuple(t["q"].shape) == (rows, HEADS, DIM),
            "only actual 3D query captures are supported",
        )
        _require(t["q"].numel() == rows * HEADS * DIM, "unexpected query size")
        _require(
            t["q"].dtype == t["kv_pages"].dtype == torch.bfloat16,
            "unexpected payload dtype",
        )
        _require(
            t["off_output"].numel() == t["q"].numel(), "unexpected captured off output"
        )
        _require(bool(torch.isfinite(t["q"]).all()), "nonfinite captured query")
        _require(
            bool(torch.isfinite(t["off_output"]).all()), "nonfinite captured off output"
        )
        table, logical, lengths = (
            t["block_table"],
            t["logical_block_table"],
            t["seq_lens"],
        )
        _require(
            table.dtype == logical.dtype == t["history"].dtype == torch.int32,
            "metadata dtype mismatch",
        )
        _require(
            lengths.shape == t["history"].shape == (rows,), "length shape mismatch"
        )
        _require(torch.equal(t["history"] + 1, lengths), "history+1 mismatch")
        _require(
            bool((t["history"] == t["history"][0]).all()),
            "mixed histories within one captured request",
        )
        _require(
            table.shape == logical.shape and table.shape[0] == rows,
            "table shape mismatch",
        )
        _require(
            int(lengths.max()) <= m["max_seq_len"], "captured maximum length mismatch"
        )
        tokens = (
            lengths.long()[:, None] - torch.arange(table.shape[1])[None, :] * PAGE
        ).clamp(0, PAGE)
        valid = tokens > 0
        ids, pages = t["physical_page_ids"], t["kv_pages"]
        _require(
            ids.ndim == 1 and ids.numel() == ids.unique().numel(),
            "duplicate physical page IDs",
        )
        _require(
            tuple(pages.shape) == (ids.numel(), 2, KV_HEADS, PAGE, DIM),
            "page shape mismatch",
        )
        _require(
            int(ids.min()) >= 0 and int(ids.max()) < m["paged_kv_metadata"]["shape"][0],
            "physical page out of range",
        )
        _require(
            int(logical[valid].min()) >= 0 and int(logical[valid].max()) < ids.numel(),
            "logical page out of range",
        )
        _require(
            torch.equal(ids[logical[valid].long()].to(table.dtype), table[valid]),
            "logical/physical mapping mismatch",
        )
        live_tokens = torch.zeros(ids.numel(), dtype=torch.int64)
        live_tokens.scatter_reduce_(
            0, logical[valid].long(), tokens[valid], reduce="amax", include_self=True
        )
        _require(
            torch.equal(live_tokens, t["live_tokens_per_page"]),
            "live-page coverage mismatch",
        )
        key, value = _unpack_pages(pages)
        used = torch.arange(PAGE)[None, :] < live_tokens[:, None]
        _require(
            bool(torch.isfinite(key.permute(0, 2, 1, 3)[used]).all()),
            "nonfinite attended keys",
        )
        _require(
            bool(torch.isfinite(value.permute(0, 2, 1, 3)[used]).all()),
            "nonfinite attended values",
        )
        fixtures.append(CapturedFixture(m, t, _sha_file(path), key, value))
    _require(observed == set(expected), "missing selected request/layer")
    return provenance, fixtures


def _view(storage, metadata):
    return torch.as_strided(
        storage,
        tuple(metadata["shape"]),
        tuple(metadata["stride"]),
        storage_offset=metadata["storage_offset"],
    )


def _comparison_report(
    actual, expected, *, atol, rtol, l2_limit, row_ids=None, mismatch_limit=None
):
    """Record every failed element without raising or hiding another backend."""
    actual = actual.to(expected.dtype)
    rows = list(range(actual.shape[0])) if row_ids is None else list(row_ids)
    _require(len(rows) == actual.shape[0], "diagnostic row mapping mismatch")
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    close = torch.isclose(actual, expected, atol=atol, rtol=rtol) & finite
    indices = (~close).nonzero()
    actual_cpu, expected_cpu = actual.detach().cpu(), expected.detach().cpu()
    mismatches = []

    def number(value):
        value = float(value)
        return value if value == value and abs(value) != float("inf") else str(value)

    detailed_indices = indices if mismatch_limit is None else indices[:mismatch_limit]
    for row, head, dim in detailed_indices.cpu().tolist():
        observed = actual_cpu[row, head, dim]
        wanted = expected_cpu[row, head, dim]
        mismatches.append(
            {
                "coordinate": [rows[row], head, dim],
                "actual": number(observed),
                "expected": number(wanted),
                "signed_error": number(observed.double() - wanted.double()),
                "allowed_error": number(atol + rtol * wanted.abs()),
            }
        )
    all_finite = bool(finite.all())
    metrics = error_metrics(actual, expected) if all_finite else None
    max_error = None
    if all_finite:
        difference = (actual.double() - expected.double()).abs()
        index = int(difference.flatten().argmax())
        row, remainder = divmod(index, actual.shape[1] * actual.shape[2])
        head, dim = divmod(remainder, actual.shape[2])
        max_error = {
            "coordinate": [rows[row], head, dim],
            "actual": number(actual_cpu[row, head, dim]),
            "expected": number(expected_cpu[row, head, dim]),
            "absolute_error": metrics["max_abs"],
        }
    l2_pass = (
        None
        if l2_limit is None
        else bool(metrics is not None and metrics["relative_l2"] <= l2_limit)
    )
    return {
        "metrics": metrics,
        "actual_finite": finite_summary(actual),
        "expected_finite": finite_summary(expected),
        "max_error": max_error,
        "thresholds": {"atol": atol, "rtol": rtol, "relative_l2": l2_limit},
        "elementwise_pass": len(indices) == 0,
        "relative_l2_pass": l2_pass,
        "thresholds_pass": len(indices) == 0 and l2_pass is not False,
        "mismatch_count": len(indices),
        "mismatch_rows": sorted(
            rows[row] for row in indices[:, 0].unique().cpu().tolist()
        ),
        "mismatch_details_truncated": len(detailed_indices) < len(indices),
        "mismatches": mismatches,
    }


def _prefix_reference(case, cuts, dtype):
    """Independent dense attention over each row's exact captured prefix cut."""
    outputs, lses = [], []
    for start in range(0, case.q.shape[0], 16):
        rows = list(range(start, min(start + 16, case.q.shape[0])))
        lengths = cuts[rows].to(device="cuda", dtype=torch.long)
        maximum = int(lengths.max())
        pages = case.logical_table[rows, : (maximum + PAGE - 1) // PAGE].cuda().long()
        key = case.k[pages].permute(0, 2, 1, 3, 4).flatten(2, 3)[:, :, :maximum]
        value = case.v[pages].permute(0, 2, 1, 3, 4).flatten(2, 3)[:, :, :maximum]
        key = key.repeat_interleave(HEADS // KV_HEADS, dim=1).to(dtype)
        value = value.repeat_interleave(HEADS // KV_HEADS, dim=1).to(dtype)
        logits = torch.einsum("bhd,bhld->bhl", case.q[rows].to(dtype), key) * DIM**-0.5
        mask = torch.arange(maximum, device="cuda")[None, :] >= lengths[:, None]
        logits.masked_fill_(mask[:, None, :], -float("inf"))
        outputs.append(torch.einsum("bhl,bhld->bhd", logits.softmax(-1), value))
        lses.append(logits.logsumexp(-1))
    return torch.cat(outputs), torch.cat(lses)


class _TailBoundaryRecorder:
    """Retain a real production launch, replacing only its two prefix inputs."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.records = []

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            if hasattr(self.kernel, "launch_kwargs"):
                kwargs = self.kernel.launch_kwargs(kwargs)
            self.records.append((grid, args, dict(kwargs), launch))
            return launch(*args, **kwargs)

        return run

    def parameters(self):
        _require(len(self.records) == 1, "expected exactly one production tail launch")
        _, args, kwargs, _ = self.records[0]
        return {**dict(zip(self.kernel.arg_names, args)), **kwargs}

    def replay(self, prefix_output, prefix_lse):
        _require(len(self.records) == 1, "ambiguous production tail launch")
        _, original_args, original_kwargs, launch = self.records[0]
        args, kwargs = list(original_args), dict(original_kwargs)
        for name, tensor in (("PART_OUT", prefix_output), ("PART_LSE", prefix_lse)):
            index = self.kernel.arg_names.index(name)
            if index < len(args):
                args[index] = tensor
            else:
                _require(name in kwargs, "missing prefix launch argument")
                kwargs[name] = tensor
        launch(*args, **kwargs)
        return self.parameters()["OUT"].clone()

    def describe(self):
        parameters = {}
        for name, value in self.parameters().items():
            if isinstance(value, torch.Tensor):
                parameters[name] = {
                    "shape": list(value.shape),
                    "stride": list(value.stride()),
                    "dtype": str(value.dtype),
                    "storage_offset": value.storage_offset(),
                }
            else:
                parameters[name] = value
        return {
            "kernel": self.kernel.__name__,
            "grid": list(self.records[0][0]),
            "parameters": parameters,
            "replaced_arguments": ["PART_OUT", "PART_LSE"],
        }


class RealCase:
    def __init__(self, fixture, arena, *, permute=False):
        m, t = fixture.manifest, fixture.tensors
        self.fixture = fixture
        self.variant = "equivalent_page_and_row_permutation" if permute else "original"
        self.seed = None
        qmeta = m["query_metadata"]
        qstorage = torch.empty(
            qmeta["storage_nbytes"] // 2, device="cuda", dtype=torch.bfloat16
        )
        self.raw_q = _view(qstorage, qmeta)
        _require(
            self.raw_q.data_ptr() % 256 == qmeta["data_ptr_mod_256"],
            "query alignment differs",
        )
        order = torch.arange(m["rows"])
        if permute:
            order = order.roll(1)
        self.raw_q.copy_(t["q"][order])
        _require(
            self.raw_q.shape == (m["rows"], HEADS, DIM),
            "only actual 3D query captures are supported",
        )
        self.q = self.raw_q
        self.kv = _view(arena, m["paged_kv_metadata"])
        _require(
            self.kv.data_ptr() % 256 == m["paged_kv_metadata"]["data_ptr_mod_256"],
            "KV alignment differs",
        )
        self.physical_ids = t["physical_page_ids"].clone()
        if permute:
            self.physical_ids = self.physical_ids.roll(1)
        self.packed = t["kv_pages"].cuda()
        self.kv.index_copy_(0, self.physical_ids.cuda().long(), self.packed)
        self.table = t["block_table"][order].clone()
        self.logical_table = t["logical_block_table"][order].clone()
        if permute:
            valid = (
                torch.arange(self.table.shape[1])[None, :] * PAGE
                < t["seq_lens"][order, None]
            )
            self.table[valid] = self.physical_ids[self.logical_table[valid].long()].to(
                self.table.dtype
            )
        self.history = t["history"][order].clone()
        self.k, self.v = fixture.key.cuda(), fixture.value.cuda()
        self.boundaries = tuple(m["request_boundaries"])
        self.request_histories = (int(self.history[0]),)
        self.high_address = m["minimum_live_byte_offset"] >= (1 << 31)
        self.boundary_address = False
        self.captured_off = t["off_output"][order].cuda().view_as(self.q)

    def describe(self):
        m = self.fixture.manifest
        return {
            "fixture": "real_model_beam_capture",
            "fixture_variant": self.variant,
            "request_ordinal": m["request_ordinal"],
            "layer_idx": m["layer_idx"],
            "manifest_sha256": self.fixture.manifest_sha256,
            "provenance_sha256": m["provenance_sha256"],
            "rows": self.q.shape[0],
            "request_boundaries": self.boundaries,
            "history_lengths": self.request_histories,
            "original_query_shape": m["query_metadata"]["shape"],
            "original_query_stride": m["query_metadata"]["stride"],
            "query_stride": self.q.stride(),
            "query_storage_offset": self.q.storage_offset(),
            "kv_span_bytes": self.kv.numel() * self.kv.element_size(),
            "replay_kv_storage_offset": self.kv.storage_offset(),
            "original_kv_storage_offset": m["paged_kv_metadata"]["storage_offset"],
            "replay_kv_storage_nbytes": self.kv.untyped_storage().nbytes(),
            "kv_cache_mem_mb_configured": m["kv_cache_mem_mb_configured"],
            "max_seq_len": m["max_seq_len"],
            "configured_max_seq_len": m["configured_max_seq_len"],
            "real_scheduler_batch": False,
        }


class RealAttentionAB(AttentionAB):
    def __init__(self, case):
        super().__init__(case)
        self.native_output_observations = []
        # Match the actual eager off invocation, not the synthetic 4096-wide
        # graph workspace. This also fixes pa_decode_gluon's partition count.
        self.off_op.enable_cuda_graph = False
        self.params.max_seq_len = case.fixture.manifest["max_seq_len"]

    def off(self):
        return self.off_op.forward(self.case.raw_q, self.cache, self.params).view_as(
            self.case.q
        )

    def on(self):
        # Observe the actual native API boundary without converting its output.
        # FP32 workspace allocation is shared with synthetic tests in AttentionAB.
        import aiter

        original = aiter.mha_batch_prefill_func

        def observe(*args, **kwargs):
            result = original(*args, **kwargs)
            provided, returned = kwargs.get("out"), result[0]
            self.native_output_observations.append(
                {
                    "provided_out_dtype": (
                        str(provided.dtype)
                        if isinstance(provided, torch.Tensor)
                        else None
                    ),
                    "returned_out_dtype": str(returned.dtype),
                    "provided_out_is_workspace": isinstance(provided, torch.Tensor)
                    and provided.data_ptr() == self.on_op.prefix_output.data_ptr(),
                    "returned_out_is_provided_storage": isinstance(
                        provided, torch.Tensor
                    )
                    and returned.data_ptr() == provided.data_ptr(),
                    "provided_out_shape": (
                        list(provided.shape)
                        if isinstance(provided, torch.Tensor)
                        else None
                    ),
                    "returned_out_shape": list(returned.shape),
                    "test_side_output_cast": False,
                }
            )
            return result

        with mock.patch.object(aiter, "mha_batch_prefill_func", side_effect=observe):
            return super().on()

    def native_output_record(self):
        requested = self.native_prefix_fp32_requested
        expected = str(torch.float32 if requested else torch.bfloat16)
        observations = self.native_output_observations
        valid = bool(observations) and str(self.on_op.prefix_output.dtype) == expected
        valid = valid and all(
            item["provided_out_dtype"] == item["returned_out_dtype"] == expected
            and item["provided_out_is_workspace"]
            and item["returned_out_is_provided_storage"]
            and item["provided_out_shape"]
            == item["returned_out_shape"]
            == list(self.case.q.shape)
            for item in observations
        )
        return {
            "FP32_O_requested": requested,
            "requested_prefix_dtype": expected,
            "allocated_prefix_dtype": str(self.on_op.prefix_output.dtype),
            "native_calls": list(observations),
            "native_output_contract_valid": valid,
            "scope": "Actual native out buffer and returned storage observed; no test-side output cast",
        }

    def update(self, new):
        _require(
            new.fixture.graph_key() == self.case.fixture.graph_key(),
            "incompatible graph fixture",
        )
        super().update(new)
        self.case.fixture, self.case.variant = new.fixture, new.variant
        self.case.captured_off, self.case.packed = new.captured_off, new.packed


class MktCascadeRealFixtureTest(MktCascadeTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        path = os.getenv("MKT_CASCADE_FIXTURE_DIR")
        if not path:
            raise RuntimeError(
                "MKT_CASCADE_FIXTURE_DIR is required; missing input is not PASS"
            )
        cls.provenance, cls.fixtures = load_fixtures(Path(path))
        backing_sizes = {
            f.manifest["paged_kv_metadata"]["storage_nbytes"] for f in cls.fixtures
        }
        _require(
            len(backing_sizes) == 1,
            "mixing distinct backing allocations requires separate runs",
        )
        cls.backing_bytes = backing_sizes.pop()
        from rtp_llm.models_py.modules.factory.attention.rocm_impl import (
            aiter as off_module,
        )

        expected_hash = cls.provenance["source_sha256"][
            "models_py/modules/factory/attention/rocm_impl/aiter.py"
        ]
        _require(
            _sha_file(Path(off_module.__file__)) == expected_hash,
            "captured off source revision differs",
        )
        for fixture in cls.fixtures:
            runtime = fixture.manifest["runtime"]
            _require(
                str(torch.__version__) == runtime["torch"],
                "captured Torch runtime differs",
            )
            _require(
                torch.version.hip == runtime["rocm"], "captured ROCm runtime differs"
            )
            for package in ("aiter", "triton"):
                _require(
                    _version(package) == runtime[package],
                    "captured attention runtime differs",
                )
            properties = torch.cuda.get_device_properties(torch.cuda.current_device())
            _require(
                properties.name == runtime["device_name"], "captured GPU model differs"
            )
            _require(
                getattr(properties, "gcnArchName", None) == runtime["gcn_arch"],
                "captured GPU architecture differs",
            )

    def _arena(self):
        return torch.empty(self.backing_bytes // 2, dtype=torch.bfloat16, device="cuda")

    def _captured_off_check(self, case, off, label):
        metrics = error_metrics(off, case.captured_off)
        exact = torch.equal(
            off.contiguous().view(torch.uint8),
            case.captured_off.contiguous().view(torch.uint8),
        )
        emit(
            label + "_captured_off",
            {
                **case.describe(),
                "off_vs_captured": metrics,
                "bitwise_equal": exact,
                "required_atol": 0,
                "required_rtol": 0,
                "off_mode": "same_eager_entry_and_max_seq_len",
            },
        )
        torch.testing.assert_close(off, case.captured_off, atol=0, rtol=0)
        self.assertTrue(exact, "captured off differs at the byte level")
        return metrics

    def _accuracy(self, ab, off, on):
        metrics = assert_accuracy(self, ab, off, on)
        selected = metrics["fp64_rows"]
        expected = reference(ab.case, selected, torch.float64)
        for name, actual in (
            ("off_vs_fp64", off[selected]),
            ("on_vs_fp64", on[selected]),
        ):
            torch.testing.assert_close(actual.double(), expected, atol=ATOL, rtol=RTOL)
            metrics[name] = error_metrics(actual, expected)
            self.assertLessEqual(metrics[name]["relative_l2"], RELATIVE_L2_LIMIT)
        return metrics

    def test_integrated_fp32_route(self):
        """Validate the service route without test-side kernel/library swaps."""
        self.assertEqual(os.getenv("RTP_ROCM_CASCADE_FP32"), "1")
        self.assertNotEqual(os.getenv("MKT_CASCADE_NATIVE_FP32_O"), "1")
        self.assertEqual(
            os.getenv("MKT_CASCADE_TAIL_VARIANT", "production"), "production"
        )
        arena = self._arena()
        anchor = max(
            self.fixtures,
            key=lambda f: f.manifest["paged_kv_metadata"]["storage_offset"],
        )
        case = RealCase(anchor, arena)
        with mock.patch.dict(os.environ, {"RTP_ROCM_CASCADE_FP32": "0"}):
            legacy = RealAttentionAB(case)
        before = legacy.on().clone()
        ab = RealAttentionAB(case)
        self.assertIsNotNone(ab.on_op.native_fp32_module)
        self.assertEqual(ab.on_op.prefix_output.dtype, torch.float32)
        graph, output = capture(ab.on)
        self.assertTrue(torch.equal(before, legacy.on()))
        addresses = (
            ab.on_op.prefix_output.data_ptr(),
            output.data_ptr(),
            case.q.data_ptr(),
            case.kv.data_ptr(),
        )
        for step, (fixture, permute) in enumerate(
            [(f, False) for f in self.fixtures]
            + [(self.fixtures[-1], True), (self.fixtures[0], False)]
        ):
            new = RealCase(fixture, arena, permute=permute)
            ab.update(new)
            output.fill_(float("nan"))
            graph.replay()
            actual = output.clone()
            eager = ab.on().clone()
            self.assertTrue(
                torch.equal(actual, eager), "Graph differs from its own eager route"
            )
            self.assertEqual(
                addresses,
                (
                    ab.on_op.prefix_output.data_ptr(),
                    output.data_ptr(),
                    case.q.data_ptr(),
                    case.kv.data_ptr(),
                ),
            )
            fp32, fp64 = reference(ab.case), reference(ab.case, dtype=torch.float64)
            reports = {
                "vs_fp32": _comparison_report(
                    actual, fp32, atol=ATOL, rtol=RTOL, l2_limit=RELATIVE_L2_LIMIT
                ),
                "vs_fp64": _comparison_report(
                    actual.double(),
                    fp64,
                    atol=ATOL,
                    rtol=RTOL,
                    l2_limit=RELATIVE_L2_LIMIT,
                ),
                "oracle": _comparison_report(
                    fp32.double(),
                    fp64,
                    atol=REFERENCE_ATOL,
                    rtol=REFERENCE_RTOL,
                    l2_limit=None,
                ),
            }
            emit(
                "integrated_fp32_route_" + str(step),
                {
                    **ab.case.describe(),
                    "reports": reports,
                    "graph_own_eager_exact": True,
                    "workspace_addresses_stable": True,
                    "module": ab.on_op.native_fp32_module.__name__,
                    "module_sha256": _sha_file(
                        Path(ab.on_op.native_fp32_module.__file__)
                    ),
                    "prefix_dtype": str(ab.on_op.prefix_output.dtype),
                    "tail": ab.on_op.tail_kernel.__name__,
                    "tail_num_warps": 1,
                    "legacy_module_coexistence_exact": True,
                },
            )
            self.assertTrue(
                all(report["thresholds_pass"] for report in reports.values())
            )

    def test_off_gluon_boundary_diagnostic(self):
        """Collect controlled off boundaries; original accuracy gates stay separate."""
        from rtp_llm.models_py.triton_kernels.attention.test.diagnose_beam_search_cascade_mkt_off_boundary import (
            diagnose_off_boundaries,
        )

        arena = self._arena()
        for fixture in self.fixtures:
            with self.subTest(fixture=fixture.name):
                case = RealCase(fixture, arena)
                ab = RealAttentionAB(case)
                report = diagnose_off_boundaries(
                    ab,
                    captured_off=case.captured_off,
                    full_reference=reference,
                    comparison_report=_comparison_report,
                    emit=lambda record: emit(
                        "real_off_boundary_" + fixture.name,
                        {
                            **case.describe(),
                            **record,
                            "end_to_end_model_accuracy": "NOT_RUN",
                        },
                    ),
                )
                self.assertTrue(report["diagnostic_controls_pass"])
                del report, ab, case

    def test_off_gluon_actual_partial_diagnostic(self):
        """Observe partial O before BF16 rounding with byte-exact controls."""
        from rtp_llm.models_py.triton_kernels.attention.test.diagnose_beam_search_cascade_mkt_off_boundary import (
            diagnose_off_boundaries,
        )
        from rtp_llm.models_py.triton_kernels.attention.test.off_precast_candidate.pa_decode_gluon_precast import (
            paged_attention_decode_v2_gluon_dot_kernel,
        )

        arena = self._arena()
        for fixture in self.fixtures:
            with self.subTest(fixture=fixture.name):
                case = RealCase(fixture, arena)
                ab = RealAttentionAB(case)
                report = diagnose_off_boundaries(
                    ab,
                    captured_off=case.captured_off,
                    full_reference=reference,
                    comparison_report=_comparison_report,
                    precast_kernel=paged_attention_decode_v2_gluon_dot_kernel,
                    emit=lambda record: emit(
                        "real_off_actual_partial_" + fixture.name,
                        {
                            **case.describe(),
                            **record,
                            "end_to_end_model_accuracy": "NOT_RUN",
                        },
                    ),
                )
                self.assertTrue(report["diagnostic_controls_pass"])
                del report, ab, case

    def test_off_gluon_probability_diagnostic(self):
        """Observe actual P conversion after full stage3 O32 fidelity checks."""
        from rtp_llm.models_py.triton_kernels.attention.test.diagnose_beam_search_cascade_mkt_off_boundary import (
            diagnose_off_boundaries,
        )
        from rtp_llm.models_py.triton_kernels.attention.test.off_precast_candidate.pa_decode_gluon_precast import (
            paged_attention_decode_v2_gluon_dot_kernel as precast_kernel,
        )
        from rtp_llm.models_py.triton_kernels.attention.test.off_probability_candidate.pa_decode_gluon_probabilities import (
            paged_attention_decode_v2_gluon_dot_kernel as probability_kernel,
        )

        arena = self._arena()
        for fixture in self.fixtures:
            with self.subTest(fixture=fixture.name):
                case = RealCase(fixture, arena)
                ab = RealAttentionAB(case)
                report = diagnose_off_boundaries(
                    ab,
                    captured_off=case.captured_off,
                    full_reference=reference,
                    comparison_report=_comparison_report,
                    precast_kernel=precast_kernel,
                    probability_kernel=probability_kernel,
                    emit=lambda record: emit(
                        "real_off_probability_" + fixture.name,
                        {
                            **case.describe(),
                            **record,
                            "end_to_end_model_accuracy": "NOT_RUN",
                        },
                    ),
                )
                self.assertTrue(report["diagnostic_controls_pass"])
                del report, ab, case

    def test_partial_prefix_boundary_diagnostic(self):
        """Test captured-instance cut714 and invalid KV independently of lineage wiring."""
        from rtp_llm.models_py.triton_kernels.attention.test.diagnose_beam_search_cascade_mkt_partial_prefix import (
            Bindings,
            run_partial_prefix_diagnostic,
        )
        from rtp_llm.models_py.triton_kernels.attention.test.test_beam_search_cascade_mkt import (
            tail_kernels,
        )

        # Explicit annotation is limited to the three captured single-request
        # instances. The helper verifies their shared partial-page bytes.
        lineage = {
            fixture.name: {
                "kind": "explicit_captured_fixture_lineage_not_production_metadata",
                "manifest_sha256": fixture.manifest_sha256,
                "request_ordinal": 0,
                "request_boundaries": [0, 500],
                "original_prompt_length": 714,
                "valid_for_captured_instance_only": True,
            }
            for fixture in self.fixtures
        }
        bindings = Bindings(
            cascade_module=cascade_mod,
            generic_tail=PRODUCTION_TAIL_KERNEL,
            best_tail=tail_kernels()["layout_w1"],
            tail_recorder_factory=_TailBoundaryRecorder,
            prefix_reference=_prefix_reference,
            full_reference=reference,
            loaded_native_modules=_loaded_native_modules,
        )
        # The helper owns the sole full-size arena, including poison restoration.
        run_partial_prefix_diagnostic(
            self.fixtures,
            lineage_by_fixture=lineage,
            expected_native_sha256=os.environ["MKT_CASCADE_EXPECTED_NATIVE_SHA256"],
            emit_record=lambda name, record: emit(
                "real_partial_prefix_" + name, record
            ),
            case_factory=RealCase,
            ab_factory=RealAttentionAB,
            bindings=bindings,
        )

    def test_partial_prefix_dynamic_graph(self):
        """Gate manual cut704/714 across changing real inputs and invalid KV."""
        from rtp_llm.models_py.triton_kernels.attention.test import (
            diagnose_beam_search_cascade_mkt_partial_prefix as partial,
        )
        from rtp_llm.models_py.triton_kernels.attention.test.diagnose_beam_search_cascade_mkt_partial_graph import (
            run_partial_prefix_graph_gate,
        )
        from rtp_llm.models_py.triton_kernels.attention.test.test_beam_search_cascade_mkt import (
            tail_kernels,
        )

        lineage = {
            fixture.name: {
                "kind": "explicit_captured_fixture_lineage_not_production_metadata",
                "manifest_sha256": fixture.manifest_sha256,
                "request_ordinal": 0,
                "request_boundaries": [0, 500],
                "original_prompt_length": 714,
                "valid_for_captured_instance_only": True,
            }
            for fixture in self.fixtures
        }
        bindings = partial.Bindings(
            cascade_module=cascade_mod,
            generic_tail=PRODUCTION_TAIL_KERNEL,
            best_tail=tail_kernels()["layout_w1"],
            tail_recorder_factory=_TailBoundaryRecorder,
            prefix_reference=_prefix_reference,
            full_reference=reference,
            loaded_native_modules=_loaded_native_modules,
        )
        run_partial_prefix_graph_gate(
            self.fixtures,
            lineage_by_fixture=lineage,
            expected_native_sha256=os.environ["MKT_CASCADE_EXPECTED_NATIVE_SHA256"],
            emit_record=emit,
            case_factory=RealCase,
            ab_factory=RealAttentionAB,
            bindings=bindings,
            partial_helpers=partial,
            capture_fn=capture,
        )

    def test_eager_real_layers(self):
        arena = self._arena()
        for fixture in self.fixtures:
            with self.subTest(fixture=fixture.name):
                case = RealCase(fixture, arena)
                self.assertEqual(
                    case.kv.storage_offset(),
                    fixture.manifest["paged_kv_metadata"]["storage_offset"],
                )
                self.assertEqual(
                    case.raw_q.stride(),
                    tuple(fixture.manifest["query_metadata"]["stride"]),
                )
                self.assertEqual(
                    case.raw_q.storage_offset(),
                    fixture.manifest["query_metadata"]["storage_offset"],
                )
                self.assertEqual(
                    case.kv.stride(),
                    tuple(fixture.manifest["paged_kv_metadata"]["stride"]),
                )
                self.assertEqual(case.kv.untyped_storage().nbytes(), self.backing_bytes)
                ab = RealAttentionAB(case)
                off, on = ab.off().clone(), ab.on().clone()
                restored = self._captured_off_check(
                    case, off, "real_eager_" + fixture.name
                )
                metrics = self._accuracy(ab, off, on)
                emit(
                    "real_eager_" + fixture.name,
                    {
                        **case.describe(),
                        "cascade_prefix": ab.prefix_path(),
                        "off_vs_captured": restored,
                        "metrics": metrics,
                        "end_to_end_model_accuracy": "NOT_RUN",
                    },
                )
                del ab, case, off, on

    def test_diagnostic_real_layers(self):
        """Evidence collection only; original numerical acceptance stays intact.

        Select independently with unittest -k diagnostic_real_layers. All
        backends run on identical real tensors and are cloned before another
        variant reuses the Cascade workspace. Assertions follow all recording.
        """
        from rtp_llm.models_py.triton_kernels.attention import (
            beam_search_cascade_tail_candidate as candidate,
        )

        kernels = {
            "production": PRODUCTION_TAIL_KERNEL,
            "common": candidate._private_tail_merge_common,
            "layout": candidate._private_tail_merge_layout,
        }
        _require(
            "AITER_AOT_IMPORT" not in os.environ,
            "diagnosis requires AITER_AOT_IMPORT unset",
        )
        build_environment = {
            name: os.getenv(name)
            for name in (
                "AITER_JIT_DIR",
                "CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT",
                "GPU_ARCHS",
                "CU_NUM",
                "AITER_AOT_IMPORT",
                "MKT_CASCADE_EXPECTED_NATIVE_SHA256",
            )
        }
        arena = self._arena()
        integrity = []
        for fixture in self.fixtures:
            case = RealCase(fixture, arena)
            ab = RealAttentionAB(case)
            outputs = {"off": ab.off().clone()}
            for name, kernel in kernels.items():
                with mock.patch.object(cascade_mod, "_private_tail_merge", kernel):
                    outputs[name] = ab.on().clone()
            fp32 = reference(case)
            fp32_reports = {
                name: _comparison_report(
                    output,
                    fp32,
                    atol=ATOL,
                    rtol=RTOL,
                    l2_limit=RELATIVE_L2_LIMIT,
                )
                for name, output in outputs.items()
            }
            ab_reports = {
                name: _comparison_report(
                    output.float(),
                    outputs["off"].float(),
                    atol=AB_ATOL,
                    rtol=AB_RTOL,
                    l2_limit=AB_RELATIVE_L2_LIMIT,
                )
                for name, output in outputs.items()
                if name != "off"
            }
            variant_reports = {}
            production_failed = {
                tuple(item["coordinate"])
                for item in fp32_reports["production"]["mismatches"]
            }
            for name in ("common", "layout"):
                candidate_failed = {
                    tuple(item["coordinate"])
                    for item in fp32_reports[name]["mismatches"]
                }
                variant_reports[name] = {
                    "vs_production_exact": _comparison_report(
                        outputs[name].float(),
                        outputs["production"].float(),
                        atol=0,
                        rtol=0,
                        l2_limit=None,
                    ),
                    "new_fp32_failure_coordinates": sorted(
                        candidate_failed - production_failed
                    ),
                    "resolved_fp32_failure_coordinates": sorted(
                        production_failed - candidate_failed
                    ),
                    "shared_fp32_failure_coordinates": sorted(
                        candidate_failed & production_failed
                    ),
                }
            # Include every row failing any numerical path, plus the original
            # endpoint/midpoint oracle rows. A global L2 failure adds all rows.
            failing_rows = set()
            for report in (*fp32_reports.values(), *ab_reports.values()):
                failing_rows.update(report["mismatch_rows"])
                if report["relative_l2_pass"] is False:
                    failing_rows.update(range(case.q.shape[0]))
            oracle_rows = sorted(
                failing_rows | {0, case.q.shape[0] // 2, case.q.shape[0] - 1}
            )
            fp64 = reference(case, oracle_rows, torch.float64)
            oracle_report = _comparison_report(
                fp32[oracle_rows].double(),
                fp64,
                atol=REFERENCE_ATOL,
                rtol=REFERENCE_RTOL,
                l2_limit=None,
                row_ids=oracle_rows,
            )
            fp64_reports = {
                name: _comparison_report(
                    output[oracle_rows].double(),
                    fp64,
                    atol=ATOL,
                    rtol=RTOL,
                    l2_limit=RELATIVE_L2_LIMIT,
                    row_ids=oracle_rows,
                )
                for name, output in outputs.items()
            }
            captured_exact = torch.equal(
                outputs["off"].contiguous().view(torch.uint8),
                case.captured_off.contiguous().view(torch.uint8),
            )
            captured_report = _comparison_report(
                outputs["off"].float(),
                case.captured_off.float(),
                atol=0,
                rtol=0,
                l2_limit=0,
            )
            native_modules = _loaded_native_modules()
            expected_native = os.getenv("MKT_CASCADE_EXPECTED_NATIVE_SHA256")
            native_valid = (
                len(native_modules) == 1 and native_modules[0]["sha256"] is not None
            )
            if expected_native:
                native_valid = (
                    native_valid and native_modules[0]["sha256"] == expected_native
                )
            emit(
                "real_diagnostic_" + fixture.name,
                {
                    **case.describe(),
                    "classification": "numerical_diagnostic_not_acceptance",
                    "cascade_prefix": ab.prefix_path(),
                    "native_prefix_output": ab.native_output_record(),
                    "fixture_tensor_sha256": {
                        name: fixture.manifest["tensors"][name]["sha256"]
                        for name in ("q", "kv_pages", "block_table", "history")
                    },
                    "off_vs_captured": captured_report,
                    "captured_off_bitwise_equal": captured_exact,
                    "backend_vs_fp32_all_rows": fp32_reports,
                    "cascade_vs_off_all_rows": ab_reports,
                    "candidate_vs_production": variant_reports,
                    "all_failed_rows": sorted(failing_rows),
                    "fp64_rows": oracle_rows,
                    "fp32_vs_fp64": oracle_report,
                    "backend_vs_fp64": fp64_reports,
                    "output_sha256": {
                        name: _tensor_hash(value.cpu())
                        for name, value in outputs.items()
                    },
                    "build_environment": build_environment,
                    "loaded_native_modules": native_modules,
                    "rounding_evidence": "Use mapped shared-object hash and its build receipt; environment alone does not establish rounding",
                    "backend_output_acceptance": "NOT_ASSERTED_BY_DIAGNOSTIC",
                    "warning": "Exit success certifies evidence collection and fixture/oracle integrity only; original accuracy failures remain failures",
                    "end_to_end_model_accuracy": "NOT_RUN",
                },
            )
            integrity.append(
                (
                    fixture.name,
                    captured_exact,
                    oracle_report["thresholds_pass"],
                    native_valid,
                    ab.native_output_record()["native_output_contract_valid"],
                )
            )
            del ab, case, outputs, fp32, fp64
        for (
            name,
            captured_exact,
            oracle_valid,
            native_valid,
            native_output_valid,
        ) in integrity:
            self.assertTrue(captured_exact, name + ": captured off replay differs")
            self.assertTrue(oracle_valid, name + ": FP32/FP64 oracle differs")
            self.assertTrue(
                native_valid,
                name + ": actual native shared-object provenance is ambiguous",
            )
            self.assertTrue(
                native_output_valid,
                name + ": requested native prefix output dtype/storage differs",
            )

    def test_prefix_boundary_diagnostic(self):
        """Counterfactual prefix boundaries; no implementation acceptance claim."""
        selected = [
            fixture for fixture in self.fixtures if fixture.manifest["layer_idx"] == 27
        ]
        _require(
            len(selected) == 1, "prefix diagnosis requires one actual layer-27 fixture"
        )
        _require(
            "AITER_AOT_IMPORT" not in os.environ,
            "diagnosis requires AITER_AOT_IMPORT unset",
        )
        fixture = selected[0]
        arena = self._arena()
        case = RealCase(fixture, arena)
        ab = RealAttentionAB(case)
        off = ab.off().clone()
        recorder = _TailBoundaryRecorder(PRODUCTION_TAIL_KERNEL)
        with mock.patch.object(cascade_mod, "_private_tail_merge", recorder):
            original_on = ab.on().clone()
        parameters = recorder.parameters()
        _require(
            parameters["NATIVE_PREFIX"] and parameters["SPLITS"] == 1,
            "expected actual native prefix launch",
        )
        cuts = parameters["CUTS"].cpu().clone()
        lengths = parameters["LENGTHS"].cpu().clone()
        _require(
            bool((cuts == 704).all()),
            "this controlled diagnosis requires actual 704-token cuts",
        )
        _require(
            torch.equal(lengths, case.history + 1),
            "actual tail lengths differ from captured attention",
        )
        native_output = parameters["PART_OUT"].clone()
        native_lse = parameters["PART_LSE"].clone()
        _require(
            native_output.shape == case.q.shape and native_output.is_contiguous(),
            "unexpected native prefix output layout",
        )
        _require(
            native_lse.shape == (HEADS, case.q.shape[0]) and native_lse.is_contiguous(),
            "unexpected native natural-log LSE layout",
        )
        prefix32, prefix_lse32 = _prefix_reference(case, cuts, torch.float32)
        prefix64, prefix_lse64 = _prefix_reference(case, cuts, torch.float64)
        # Both O intervention arms share the same FP32 oracle source. This
        # avoids confounding FP64 -> BF16 with FP64 -> FP32 -> BF16 rounding.
        prefix_bf16_rn = prefix32.to(torch.bfloat16).contiguous()
        cast_matches_cpu_rne = torch.equal(
            prefix_bf16_rn.cpu().view(torch.uint8),
            prefix32.cpu().to(torch.bfloat16).view(torch.uint8),
        )
        oracle_lse = prefix_lse32.T.contiguous()
        arms = {
            "native_O_native_LSE": (native_output, native_lse),
            "oracle_FP32_O_cast_BF16_RN_native_LSE": (prefix_bf16_rn, native_lse),
            "oracle_FP32_O_native_LSE": (prefix32.contiguous(), native_lse),
            "oracle_FP32_O_oracle_natural_log_LSE": (prefix32.contiguous(), oracle_lse),
        }
        outputs = {
            name: recorder.replay(output, lse) for name, (output, lse) in arms.items()
        }
        # Same numerical prefix values in a different pointer dtype isolate
        # dtype specialization from the B -> C value-precision intervention.
        dtype_control = recorder.replay(prefix_bf16_rn.float(), native_lse)
        dtype_control_expected = outputs["oracle_FP32_O_cast_BF16_RN_native_LSE"]
        dtype_control_exact = torch.equal(
            dtype_control.view(torch.uint8),
            dtype_control_expected.view(torch.uint8),
        )
        dtype_control_report = _comparison_report(
            dtype_control.float(),
            dtype_control_expected.float(),
            atol=0,
            rtol=0,
            l2_limit=None,
        )
        original_exact = torch.equal(
            outputs["native_O_native_LSE"].view(torch.uint8),
            original_on.view(torch.uint8),
        )
        full32 = reference(case)
        full64 = reference(case, dtype=torch.float64)

        def compare(actual, expected, *, oracle=False, prefix=False):
            return _comparison_report(
                actual,
                expected,
                atol=REFERENCE_ATOL if oracle else ATOL,
                rtol=REFERENCE_RTOL if oracle else RTOL,
                l2_limit=None if oracle else RELATIVE_L2_LIMIT,
                mismatch_limit=64 if prefix else None,
            )

        all_outputs = {"off": off, **outputs}
        full32_reports = {
            name: compare(output, full32) for name, output in all_outputs.items()
        }
        full64_reports = {
            name: compare(output.double(), full64)
            for name, output in all_outputs.items()
        }
        prefix_output_reports = {
            "native_vs_fp32": compare(native_output, prefix32, prefix=True),
            "native_vs_fp64": compare(native_output.double(), prefix64, prefix=True),
            "fp32_vs_fp64": compare(
                prefix32.double(), prefix64, oracle=True, prefix=True
            ),
            "native_vs_oracle_FP32_cast_BF16_RN": _comparison_report(
                native_output.float(),
                prefix_bf16_rn.float(),
                atol=0,
                rtol=0,
                l2_limit=None,
                mismatch_limit=64,
            ),
            "oracle_FP32_cast_BF16_RN_vs_fp32": compare(
                prefix_bf16_rn, prefix32, prefix=True
            ),
        }
        # _comparison_report uses [row, head, dim]; LSE has a single scalar dim.
        prefix_lse_reports = {
            "native_vs_fp32": compare(
                native_lse.T[..., None],
                prefix_lse32[..., None],
                oracle=True,
                prefix=True,
            ),
            "native_vs_fp64": compare(
                native_lse.T.double()[..., None],
                prefix_lse64[..., None],
                oracle=True,
                prefix=True,
            ),
            "fp32_vs_fp64": compare(
                prefix_lse32.double()[..., None],
                prefix_lse64[..., None],
                oracle=True,
                prefix=True,
            ),
        }
        full_oracle_report = compare(full32.double(), full64, oracle=True)
        native_failed = {
            tuple(item["coordinate"])
            for item in full32_reports["native_O_native_LSE"]["mismatches"]
        }
        boundary_at_native_failures = []
        for row, head, dim in sorted(native_failed):
            boundary_at_native_failures.append(
                {
                    "coordinate": [row, head, dim],
                    "prefix_O": {
                        "native": float(native_output[row, head, dim]),
                        "oracle_fp32": float(prefix32[row, head, dim]),
                        "oracle_fp64": float(prefix64[row, head, dim]),
                        "oracle_fp32_cast_bf16_rn": float(
                            prefix_bf16_rn[row, head, dim]
                        ),
                    },
                    "prefix_natural_log_LSE": {
                        "native": float(native_lse[head, row]),
                        "oracle_fp32": float(prefix_lse32[row, head]),
                        "oracle_fp64": float(prefix_lse64[row, head]),
                    },
                    "full_attention": {
                        **{
                            name: float(output[row, head, dim])
                            for name, output in all_outputs.items()
                        },
                        "oracle_fp32": float(full32[row, head, dim]),
                        "oracle_fp64": float(full64[row, head, dim]),
                        "allowed_error_fp32": float(
                            ATOL + RTOL * full32[row, head, dim].abs()
                        ),
                    },
                }
            )
        failure_changes = {}
        for name in outputs:
            failed = {
                tuple(item["coordinate"]) for item in full32_reports[name]["mismatches"]
            }
            failure_changes[name] = {
                "resolved_native_failure_coordinates": sorted(native_failed - failed),
                "remaining_native_failure_coordinates": sorted(native_failed & failed),
                "new_failure_coordinates": sorted(failed - native_failed),
            }
        native_modules = _loaded_native_modules()
        expected_native = os.getenv("MKT_CASCADE_EXPECTED_NATIVE_SHA256")
        native_valid = (
            len(native_modules) == 1 and native_modules[0]["sha256"] is not None
        )
        if expected_native:
            native_valid = (
                native_valid and native_modules[0]["sha256"] == expected_native
            )
        captured_exact = torch.equal(
            off.view(torch.uint8), case.captured_off.view(torch.uint8)
        )
        emit(
            "real_prefix_boundary_" + fixture.name,
            {
                **case.describe(),
                "classification": "counterfactual_prefix_boundary_not_acceptance",
                "production_tail_launch": recorder.describe(),
                "production_tail_source_sha256": _sha_file(Path(cascade_mod.__file__)),
                "counterfactual_prefix_inputs": {
                    name: {
                        "O_dtype": str(output.dtype),
                        "O_stride": list(output.stride()),
                        "LSE_dtype": str(lse.dtype),
                        "LSE_stride": list(lse.stride()),
                    }
                    for name, (output, lse) in arms.items()
                },
                "original_tail_replay_bitwise_equal": original_exact,
                "same_BF16_values_in_FP32_storage_control": {
                    "bitwise_equal": dtype_control_exact,
                    "comparison": dtype_control_report,
                },
                "captured_off_bitwise_equal": captured_exact,
                "prefix_token_range": [0, 704],
                "actual_attended_lengths": lengths.unique().tolist(),
                "actual_tail_lengths": (lengths - cuts).unique().tolist(),
                "prefix_O_vs_oracle": prefix_output_reports,
                "prefix_natural_log_LSE_vs_oracle": prefix_lse_reports,
                "full_attention_vs_fp32_all_rows": full32_reports,
                "full_attention_vs_fp64_all_rows": full64_reports,
                "full_fp32_vs_fp64_all_rows": full_oracle_report,
                "native_failure_coordinates": sorted(native_failed),
                "prefix_boundary_at_every_native_failure": boundary_at_native_failures,
                "failure_changes": failure_changes,
                "oracle_rows": list(range(case.q.shape[0])),
                "BF16_intervention_conversion": "same FP32 oracle O -> torch.bfloat16 round-to-nearest-even",
                "BF16_intervention_matches_CPU_RNE": cast_matches_cpu_rne,
                "LSE_intervention_layout": "natural log; FP32 [head,row] contiguous",
                "loaded_native_modules": native_modules,
                "build_environment": {
                    name: os.getenv(name)
                    for name in (
                        "AITER_JIT_DIR",
                        "CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT",
                        "GPU_ARCHS",
                        "CU_NUM",
                        "AITER_AOT_IMPORT",
                        "MKT_CASCADE_EXPECTED_NATIVE_SHA256",
                    )
                },
                "rounding_evidence": "Actual mapped shared-object hash plus build receipt, not environment alone",
                "backend_output_acceptance": "NOT_ASSERTED_BY_COUNTERFACTUAL",
                "prefix_report_detail_scope": "All prefix metrics/counts/failed rows; first 64 individual prefix mismatches plus every final native failure boundary. All final-attention mismatches retained.",
                "warning": "The fourth arm still includes tail QK/exp/PV, FP32 merging and final BF16 storage; residual errors do not identify a layout-specific cause",
                "end_to_end_model_accuracy": "NOT_RUN",
            },
        )
        self.assertTrue(
            original_exact, "captured original tail launch is not reproducible"
        )
        self.assertTrue(
            dtype_control_exact,
            "same prefix values differ under FP32 pointer specialization",
        )
        self.assertTrue(
            cast_matches_cpu_rne,
            "BF16 intervention differs from CPU round-to-nearest-even cast",
        )
        self.assertTrue(captured_exact, "captured eager off differs")
        self.assertTrue(
            native_valid, "actual native shared-object provenance is ambiguous"
        )
        self.assertTrue(
            prefix_output_reports["fp32_vs_fp64"]["thresholds_pass"],
            "prefix O oracle differs",
        )
        self.assertTrue(
            prefix_lse_reports["fp32_vs_fp64"]["thresholds_pass"],
            "prefix LSE oracle differs",
        )
        self.assertTrue(
            full_oracle_report["thresholds_pass"], "full attention oracle differs"
        )

    def test_existing_triton_prefix_diagnostic(self):
        """Observe the existing Triton route with the native-capable construction.

        This changes only the run-time byte-limit guard in the test process.
        BEAMS_PER_TILE, splits, storage and planning retain their original values;
        it is not the separate constructor path using 16-beam query partitions.
        """
        _require(
            "AITER_AOT_IMPORT" not in os.environ,
            "diagnosis requires AITER_AOT_IMPORT unset",
        )
        _require(
            os.getenv("MKT_CASCADE_FORCE_NATIVE_LARGE", "0") != "1",
            "instance guard override conflicts with route diagnosis",
        )
        owner = cascade_mod.RocmBeamSearchCascadeAttention
        original_limit = owner.NATIVE_PREFIX_BYTE_LIMIT
        arena = self._arena()
        integrity = []
        for fixture in self.fixtures:
            case = RealCase(fixture, arena)
            ab = RealAttentionAB(case)
            _require(
                "NATIVE_PREFIX_BYTE_LIMIT" not in vars(ab.on_op),
                "unexpected per-instance route override",
            )

            def construction():
                return {
                    "native_prefix_capable": ab.on_op.native_prefix,
                    "BEAMS_PER_TILE": ab.on_op.BEAMS_PER_TILE,
                    "splits": ab.on_op.splits,
                    "tiles": ab.on_op.tiles,
                    "max_pages": ab.on_op.max_pages,
                    "rows": ab.on_op.rows,
                }

            original_construction = construction()
            off = ab.off().clone()
            native_recorder = _TailBoundaryRecorder(PRODUCTION_TAIL_KERNEL)
            with mock.patch.object(cascade_mod, "_private_tail_merge", native_recorder):
                native = ab.on().clone()
            native_parameters = native_recorder.parameters()
            shared_recorder = _TailBoundaryRecorder(cascade_mod._shared_prefix)
            tail_recorder = _TailBoundaryRecorder(PRODUCTION_TAIL_KERNEL)
            with mock.patch.object(
                owner, "NATIVE_PREFIX_BYTE_LIMIT", 0
            ), mock.patch.object(
                cascade_mod, "_shared_prefix", shared_recorder
            ), mock.patch.object(
                cascade_mod, "_private_tail_merge", tail_recorder
            ):
                triton_output = ab.on().clone()
                forced_construction = construction()
            shared_parameters, tail_parameters = (
                shared_recorder.parameters(),
                tail_recorder.parameters(),
            )
            route_checks = {
                "original_native_capable_geometry": original_construction[
                    "native_prefix_capable"
                ]
                and original_construction["BEAMS_PER_TILE"] == 128
                and original_construction["splits"] == 1,
                "original_native_tail_flag": bool(native_parameters["NATIVE_PREFIX"]),
                "original_native_prefix_output_BF16": native_parameters[
                    "PART_OUT"
                ].dtype
                == torch.bfloat16,
                "existing_shared_prefix_launched_once": len(shared_recorder.records)
                == 1,
                "forced_tail_native_flag_false": not bool(
                    tail_parameters["NATIVE_PREFIX"]
                ),
                "shared_prefix_output_FP32": shared_parameters["PART_OUT"].dtype
                == torch.float32,
                "tail_prefix_input_FP32": tail_parameters["PART_OUT"].dtype
                == torch.float32,
                "shared_prefix_LSE_FP32": shared_parameters["PART_LSE"].dtype
                == torch.float32,
                "same_prefix_output_storage": shared_parameters["PART_OUT"].data_ptr()
                == tail_parameters["PART_OUT"].data_ptr(),
                "same_prefix_LSE_storage": shared_parameters["PART_LSE"].data_ptr()
                == tail_parameters["PART_LSE"].data_ptr(),
                "constructor_unchanged": original_construction
                == forced_construction
                == construction(),
                "original_guard_restored": owner.NATIVE_PREFIX_BYTE_LIMIT
                == original_limit,
            }
            full32 = reference(case)
            full64 = reference(case, dtype=torch.float64)
            outputs = {
                "off": off,
                "native_prefix_production_tail": native,
                "existing_triton_prefix_production_tail": triton_output,
            }
            fp32_reports = {
                name: _comparison_report(
                    output, full32, atol=ATOL, rtol=RTOL, l2_limit=RELATIVE_L2_LIMIT
                )
                for name, output in outputs.items()
            }
            fp64_reports = {
                name: _comparison_report(
                    output.double(),
                    full64,
                    atol=ATOL,
                    rtol=RTOL,
                    l2_limit=RELATIVE_L2_LIMIT,
                )
                for name, output in outputs.items()
            }
            ab_reports = {
                name: _comparison_report(
                    output.float(),
                    off.float(),
                    atol=AB_ATOL,
                    rtol=AB_RTOL,
                    l2_limit=AB_RELATIVE_L2_LIMIT,
                )
                for name, output in outputs.items()
                if name != "off"
            }
            oracle_report = _comparison_report(
                full32.double(),
                full64,
                atol=REFERENCE_ATOL,
                rtol=REFERENCE_RTOL,
                l2_limit=None,
            )
            captured_exact = torch.equal(
                off.view(torch.uint8), case.captured_off.view(torch.uint8)
            )
            captured_report = _comparison_report(
                off.float(), case.captured_off.float(), atol=0, rtol=0, l2_limit=0
            )
            native_modules = _loaded_native_modules()
            expected_native = os.getenv("MKT_CASCADE_EXPECTED_NATIVE_SHA256")
            native_valid = (
                len(native_modules) == 1 and native_modules[0]["sha256"] is not None
            )
            if expected_native:
                native_valid = (
                    native_valid and native_modules[0]["sha256"] == expected_native
                )
            emit(
                "real_existing_triton_" + fixture.name,
                {
                    **case.describe(),
                    "classification": "existing_route_diagnostic_not_acceptance",
                    "original_construction": original_construction,
                    "forced_route_construction": forced_construction,
                    "construction_scope": "Original native-capable BEAMS_PER_TILE=128/splits=1 construction retained; not a 16-beam constructor experiment",
                    "original_native_prefix_byte_limit": original_limit,
                    "test_forced_native_prefix_byte_limit": 0,
                    "route_checks": route_checks,
                    "native_tail_launch": {
                        **native_recorder.describe(),
                        "replaced_arguments": [],
                    },
                    "existing_shared_prefix_launch": {
                        **shared_recorder.describe(),
                        "replaced_arguments": [],
                    },
                    "existing_triton_tail_launch": {
                        **tail_recorder.describe(),
                        "replaced_arguments": [],
                    },
                    "production_source_sha256": _sha_file(Path(cascade_mod.__file__)),
                    "captured_off_bitwise_equal": captured_exact,
                    "off_vs_captured": captured_report,
                    "backend_vs_fp32_all_rows": fp32_reports,
                    "backend_vs_fp64_all_rows": fp64_reports,
                    "cascade_vs_off_all_rows": ab_reports,
                    "fp32_vs_fp64_all_rows": oracle_report,
                    "oracle_rows": list(range(case.q.shape[0])),
                    "output_sha256": {
                        name: _tensor_hash(output.cpu())
                        for name, output in outputs.items()
                    },
                    "loaded_native_modules": native_modules,
                    "build_environment": {
                        name: os.getenv(name)
                        for name in (
                            "AITER_JIT_DIR",
                            "CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT",
                            "GPU_ARCHS",
                            "CU_NUM",
                            "AITER_AOT_IMPORT",
                            "MKT_CASCADE_EXPECTED_NATIVE_SHA256",
                        )
                    },
                    "backend_output_acceptance": "NOT_ASSERTED_BY_DIAGNOSTIC",
                    "warning": "Exit success certifies recording and fixture/oracle/route integrity; reported fixed-threshold failures remain failures",
                    "end_to_end_model_accuracy": "NOT_RUN",
                },
            )
            integrity.append(
                (
                    fixture.name,
                    captured_exact,
                    oracle_report["thresholds_pass"],
                    all(route_checks.values()),
                    native_valid,
                )
            )
            del ab, case, outputs, off, native, triton_output, full32, full64
            del (
                native_recorder,
                shared_recorder,
                tail_recorder,
                native_parameters,
                shared_parameters,
                tail_parameters,
            )
        for name, captured_exact, oracle_valid, route_valid, native_valid in integrity:
            self.assertTrue(captured_exact, name + ": captured eager off differs")
            self.assertTrue(oracle_valid, name + ": FP32/FP64 oracle differs")
            self.assertTrue(
                route_valid,
                name + ": actual route or construction differs from the experiment",
            )
            self.assertTrue(
                native_valid, name + ": native shared-object provenance is ambiguous"
            )

    def test_native_fp32_candidate_acceptance(self):
        """Gate only the native-FP32-O candidate against independent oracles.

        The original off/on acceptance methods remain separate and unchanged.
        Actual off errors are reported as their own FAIL; they cannot become a
        PASS through this candidate-only gate or through captured-output equality.
        """
        _require(
            os.getenv("MKT_CASCADE_NATIVE_FP32_O") == "1",
            "candidate gate requires explicit native FP32 O",
        )
        _require(
            "AITER_AOT_IMPORT" not in os.environ,
            "candidate gate requires AITER_AOT_IMPORT unset",
        )
        arena = self._arena()
        expected_warps = {"layout_w1": 1, "layout_w4": 4, "shared_k_w1": 1}.get(
            os.getenv("MKT_CASCADE_TAIL_VARIANT", "production"), 2
        )

        def invoke_candidate(ab):
            recorder = _TailBoundaryRecorder(cascade_mod._private_tail_merge)
            with mock.patch.object(cascade_mod, "_private_tail_merge", recorder):
                output = ab.on()
            parameters = recorder.parameters()
            ab.candidate_tail_launch = {
                "kernel": recorder.kernel.__name__,
                "grid": list(recorder.records[0][0]),
                "num_warps": int(parameters["num_warps"]),
                "part_out_dtype": str(parameters["PART_OUT"].dtype),
            }
            return output

        def evaluate(ab, on, off, label, extra):
            case = ab.case
            fp32 = reference(case)
            fp64 = reference(case, dtype=torch.float64)
            on32 = _comparison_report(
                on, fp32, atol=ATOL, rtol=RTOL, l2_limit=RELATIVE_L2_LIMIT
            )
            on64 = _comparison_report(
                on.double(), fp64, atol=ATOL, rtol=RTOL, l2_limit=RELATIVE_L2_LIMIT
            )
            off32 = _comparison_report(
                off, fp32, atol=ATOL, rtol=RTOL, l2_limit=RELATIVE_L2_LIMIT
            )
            off64 = _comparison_report(
                off.double(), fp64, atol=ATOL, rtol=RTOL, l2_limit=RELATIVE_L2_LIMIT
            )
            oracle = _comparison_report(
                fp32.double(),
                fp64,
                atol=REFERENCE_ATOL,
                rtol=REFERENCE_RTOL,
                l2_limit=None,
            )
            captured_exact = torch.equal(
                off.contiguous().view(torch.uint8),
                case.captured_off.contiguous().view(torch.uint8),
            )
            native_record = ab.native_output_record()
            native_modules = _loaded_native_modules()
            expected_native = os.getenv("MKT_CASCADE_EXPECTED_NATIVE_SHA256")
            native_valid = (
                len(native_modules) == 1 and native_modules[0]["sha256"] is not None
            )
            if expected_native:
                native_valid = (
                    native_valid and native_modules[0]["sha256"] == expected_native
                )
            candidate_pass = on32["thresholds_pass"] and on64["thresholds_pass"]
            off_pass = off32["thresholds_pass"] and off64["thresholds_pass"]
            controls = {
                "captured_off_bitwise_equal": captured_exact,
                "oracle_FP32_FP64_valid": oracle["thresholds_pass"],
                "native_FP32_O_requested": native_record["FP32_O_requested"],
                "native_FP32_O_contract_valid": native_record[
                    "native_output_contract_valid"
                ],
                "actual_native_route": ab.prefix_path() == "aiter_native",
                "native_shared_object_valid": native_valid,
                "tail_launch_geometry_matches_requested": ab.candidate_tail_launch[
                    "num_warps"
                ]
                == expected_warps,
                **extra["controls"],
            }
            record = {
                **case.describe(),
                "classification": "native_FP32_O_candidate_only_acceptance",
                "tail_kernel": cascade_mod._private_tail_merge.__name__,
                "requested_tail_variant": os.getenv(
                    "MKT_CASCADE_TAIL_VARIANT", "production"
                ),
                "actual_tail_launch": ab.candidate_tail_launch,
                "candidate_numerical_gate": "PASS" if candidate_pass else "FAIL",
                "independent_off_numerical_gate": "PASS" if off_pass else "FAIL",
                "candidate_on_vs_fp32_all_rows": on32,
                "candidate_on_vs_fp64_all_rows": on64,
                "independent_off_vs_fp32_all_rows": off32,
                "independent_off_vs_fp64_all_rows": off64,
                "fp32_vs_fp64_all_rows": oracle,
                "candidate_controls": controls,
                "native_prefix_output": native_record,
                "loaded_native_modules": native_modules,
                "build_environment": {
                    name: os.getenv(name)
                    for name in (
                        "MKT_CASCADE_NATIVE_FP32_O",
                        "AITER_JIT_DIR",
                        "CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT",
                        "GPU_ARCHS",
                        "CU_NUM",
                        "AITER_AOT_IMPORT",
                        "MKT_CASCADE_EXPECTED_NATIVE_SHA256",
                    )
                },
                "execution": {
                    key: value for key, value in extra.items() if key != "controls"
                },
                "off_scope": "Independent off failure remains FAIL; original off/on acceptance methods are retained",
                "oracle_rows": list(range(case.q.shape[0])),
                "end_to_end_model_accuracy": "NOT_RUN",
            }
            emit(label, record)
            return candidate_pass, controls

        def assert_candidate(result):
            numerical_pass, controls = result
            for name, valid in controls.items():
                self.assertTrue(valid, name)
            self.assertTrue(
                numerical_pass,
                "native FP32 O candidate exceeds the unchanged FP32/FP64 thresholds",
            )

        for fixture in self.fixtures:
            with self.subTest(mode="eager", fixture=fixture.name):
                case = RealCase(fixture, arena)
                ab = RealAttentionAB(case)
                ab.on_op.prefix_output.fill_(float("nan"))
                on, off = invoke_candidate(ab).clone(), ab.off().clone()
                result = evaluate(
                    ab,
                    on,
                    off,
                    "real_native_fp32_eager_" + fixture.name,
                    {
                        "mode": "eager",
                        "prefix_buffer_poisoned_before_execution": True,
                        "controls": {
                            "original_query_stride_preserved": case.raw_q.stride()
                            == tuple(fixture.manifest["query_metadata"]["stride"]),
                            "original_KV_storage_offset_preserved": case.kv.storage_offset()
                            == fixture.manifest["paged_kv_metadata"]["storage_offset"],
                        },
                    },
                )
                assert_candidate(result)
                del ab, case, on, off

        groups = {}
        for fixture in self.fixtures:
            groups.setdefault(fixture.graph_key(), []).append(fixture)
        for group_index, fixtures in enumerate(groups.values()):
            anchor = max(
                fixtures,
                key=lambda item: item.manifest["paged_kv_metadata"]["storage_offset"],
            )
            case = RealCase(anchor, arena)
            ab = RealAttentionAB(case)
            graph, output = capture(lambda: invoke_candidate(ab))
            captured_tail_launch = dict(ab.candidate_tail_launch)

            def addresses():
                return tuple(
                    tensor.data_ptr()
                    for tensor in (
                        case.q,
                        case.kv,
                        output,
                        ab.on_op.prefix_output,
                        ab.on_op.table,
                        ab.on_op.lengths,
                        ab.on_op.row_cuts,
                        ab.on_op.cuts,
                        ab.on_op.prefix_qptr,
                        ab.on_op.prefix_indptr,
                        ab.on_op.prefix_table,
                    )
                )

            original_addresses = addresses()
            steps = [(fixture, False) for fixture in fixtures]
            steps.extend(((fixtures[-1], True), (fixtures[0], False)))
            previous_kv_hash = anchor.manifest["tensors"]["kv_pages"]["sha256"]
            for step, (fixture, permute) in enumerate(steps):
                with self.subTest(
                    mode="graph", group=group_index, step=step, fixture=fixture.name
                ):
                    previous_q, previous_table = case.q.clone(), case.table.clone()
                    new = RealCase(fixture, arena, permute=permute)
                    q_changed = not torch.equal(previous_q, new.q)
                    table_changed = not torch.equal(previous_table, new.table)
                    ab.update(new)
                    output.fill_(float("nan"))
                    ab.on_op.prefix_output.fill_(float("nan"))
                    graph.replay()
                    replay = output.clone()
                    eager, off = invoke_candidate(ab).clone(), ab.off().clone()
                    own_exact = torch.equal(
                        replay.contiguous().view(torch.uint8),
                        eager.contiguous().view(torch.uint8),
                    )
                    kv_hash = fixture.manifest["tensors"]["kv_pages"]["sha256"]
                    result = evaluate(
                        ab,
                        replay,
                        off,
                        f"real_native_fp32_graph_group{group_index}_step{step}_{fixture.name}",
                        {
                            "mode": "graph",
                            "graph_group_count": len(groups),
                            "compatible_real_layers": [
                                item.manifest["layer_idx"] for item in fixtures
                            ],
                            "graph_anchor_layer": anchor.manifest["layer_idx"],
                            "q_values_changed": q_changed,
                            "kv_values_changed": kv_hash != previous_kv_hash,
                            "physical_table_changed": table_changed,
                            "history_update_scope": "Actual captured history copied; no fabricated history changes",
                            "metadata_variant": case.variant,
                            "prefix_and_final_output_poisoned_before_replay": True,
                            "graph_vs_own_eager": _comparison_report(
                                replay.float(),
                                eager.float(),
                                atol=0,
                                rtol=0,
                                l2_limit=0,
                            ),
                            "controls": {
                                "graph_vs_own_eager_bitwise_equal": own_exact,
                                "graph_addresses_stable": addresses()
                                == original_addresses,
                                "graph_tail_launch_geometry_unchanged": ab.candidate_tail_launch
                                == captured_tail_launch,
                                "equivalent_metadata_changed_when_requested": not permute
                                or table_changed,
                            },
                        },
                    )
                    assert_candidate(result)
                    previous_kv_hash = kv_hash
                    del new, previous_q, previous_table, replay, eager, off
            del ab, case, output, graph

    def test_real_attention_performance(self):
        """Fair fixed-Graph attention measurements after per-arm real accuracy gates."""
        import statistics

        import aiter
        import triton
        import triton.testing

        from rtp_llm.models_py.triton_kernels.attention import (
            beam_search_cascade_tail_candidate as candidate,
        )
        from rtp_llm.models_py.triton_kernels.attention.test import (
            test_beam_search_cascade_mkt as helper,
        )

        _require(
            os.getenv("MKT_CASCADE_NATIVE_FP32_O") == "1",
            "performance comparison requires native FP32 O",
        )
        _require(
            os.getenv("MKT_CASCADE_TAIL_VARIANT", "production") == "production",
            "performance comparison owns its tail patches",
        )
        _require(
            os.getenv("MKT_CASCADE_FORCE_NATIVE_LARGE", "0") != "1",
            "instance byte-limit override conflicts with route comparison",
        )
        _require(
            "AITER_AOT_IMPORT" not in os.environ,
            "performance comparison requires AITER_AOT_IMPORT unset",
        )
        expected_native = os.getenv("MKT_CASCADE_EXPECTED_NATIVE_SHA256")
        _require(
            bool(expected_native),
            "performance comparison requires an expected native shared-object hash",
        )
        comparison_mode = os.getenv("MKT_CASCADE_REAL_PERF_COMPARISON", "prefix_routes")
        _require(
            comparison_mode in ("prefix_routes", "waves", "partial_prefix"),
            "unknown real performance comparison",
        )
        names = (
            "native_FP32_production",
            "native_FP32_layout",
            (
                "native_FP32_layout_w1"
                if comparison_mode == "waves"
                else "triton_FP32_layout"
            ),
        )
        kernels = {
            names[0]: PRODUCTION_TAIL_KERNEL,
            names[1]: candidate._private_tail_merge_layout,
            names[2]: (
                helper.tail_kernels()["layout_w1"]
                if comparison_mode == "waves"
                else candidate._private_tail_merge_layout
            ),
        }
        partial = None
        if comparison_mode == "partial_prefix":
            from contextlib import ExitStack

            from rtp_llm.models_py.triton_kernels.attention.test import (
                diagnose_beam_search_cascade_mkt_partial_prefix as partial,
            )

            # Isolate the cut change with the same generic tail/launch; retain
            # the measured best complete-page tail as a separate baseline.
            names = (
                "native_FP32_layout_w1_cut704",
                "native_FP32_generic_cut704",
                "native_FP32_generic_cut714",
            )
            kernels = {
                names[0]: helper.tail_kernels()["layout_w1"],
                names[1]: PRODUCTION_TAIL_KERNEL,
                names[2]: PRODUCTION_TAIL_KERNEL,
            }
            arm_cuts = dict(zip(names, (704, 704, 714)))
            cut_contexts = ExitStack()
            self.addCleanup(cut_contexts.close)
        owner = cascade_mod.RocmBeamSearchCascadeAttention
        original_limit = owner.NATIVE_PREFIX_BYTE_LIMIT
        hot_unroll, hot_samples = 64, 10
        source_paths = (
            Path(__file__),
            Path(helper.__file__),
            Path(cascade_mod.__file__),
            Path(candidate.__file__),
        )
        if partial is not None:
            source_paths += (Path(partial.__file__),)

        def source_hashes():
            return {str(path): _sha_file(path) for path in source_paths}

        def fixed_call(record):
            _, args, kwargs, launch = record
            return lambda: launch(*args, **kwargs)

        def hot_measure(graph):
            # The unrolled graph was captured once during preparation. No
            # Python model call, patch or dtype inspection occurs in the timer.
            graph.replay()
            graph.replay()
            events = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(hot_samples)
            ]
            for start, end in events:
                start.record()
                graph.replay()
                end.record()
            torch.cuda.synchronize()
            return [start.elapsed_time(end) / hot_unroll for start, end in events]

        # Read the actual driver allocation size; do_bench uses this same
        # driver factory and clears its buffer before every timed replay.
        cache_probe = triton.runtime.driver.active.get_empty_cache_for_benchmark()
        cache_clear_bytes = cache_probe.numel() * cache_probe.element_size()
        del cache_probe
        source_before = source_hashes()
        arena = self._arena()
        for fixture in self.fixtures:
            lineage = None
            if partial is not None:
                declaration = {
                    "kind": "explicit_captured_fixture_lineage_not_production_metadata",
                    "manifest_sha256": fixture.manifest_sha256,
                    "request_ordinal": 0,
                    "request_boundaries": [0, 500],
                    "original_prompt_length": 714,
                    "valid_for_captured_instance_only": True,
                }
                _, lineage = partial._validate_lineage(fixture, declaration)
            case = RealCase(fixture, arena)
            fp32 = reference(case)
            fp64 = reference(case, dtype=torch.float64)
            oracle_report = _comparison_report(
                fp32.double(),
                fp64,
                atol=REFERENCE_ATOL,
                rtol=REFERENCE_RTOL,
                l2_limit=None,
            )
            operators, prepared, preflight, routes = {}, {}, {}, {}
            shared_parameters = None

            def input_hashes():
                pages = case.kv.index_select(0, case.physical_ids.cuda().long())
                return {
                    "q": _tensor_hash(case.q.cpu()),
                    "kv_pages": _tensor_hash(pages.cpu()),
                }

            input_before = input_hashes()
            _require(
                input_before
                == {
                    name: fixture.manifest["tensors"][name]["sha256"]
                    for name in input_before
                },
                "GPU inputs differ from captured real tensors",
            )
            for name in names:
                native_route = name != "triton_FP32_layout"
                limit = original_limit if native_route else 0
                ab = RealAttentionAB(case)
                operators[name] = ab
                if partial is not None:
                    # Keep each operator's private cut buffers at its declared
                    # values throughout capture and timing; restore on failure
                    # through unittest cleanup as well as on the success path.
                    cut_contexts.enter_context(partial._manual_cut(ab, arm_cuts[name]))
                native_calls = []
                original_native = aiter.mha_batch_prefill_func

                def record_native(*args, **kwargs):
                    native_calls.append((args, dict(kwargs)))
                    return original_native(*args, **kwargs)

                shared_recorder = _TailBoundaryRecorder(cascade_mod._shared_prefix)
                tail_recorder = _TailBoundaryRecorder(kernels[name])
                with mock.patch.object(
                    owner, "NATIVE_PREFIX_BYTE_LIMIT", limit
                ), mock.patch.object(cascade_mod, "_private_tail_merge", kernels[name]):
                    # Record actual boundaries once; the original whole run
                    # connects that invocation's prefix O/LSE to its own tail.
                    with mock.patch.object(
                        aiter, "mha_batch_prefill_func", side_effect=record_native
                    ), mock.patch.object(
                        cascade_mod, "_shared_prefix", shared_recorder
                    ), mock.patch.object(
                        cascade_mod, "_private_tail_merge", tail_recorder
                    ):
                        actual = ab.on().clone()
                    tail_parameters = tail_recorder.parameters()
                    if native_route:
                        _require(
                            len(native_calls) == 1 and not shared_recorder.records,
                            "unexpected native prefix dispatch",
                        )
                        prefix_args, prefix_kwargs = native_calls[0]
                        prefix_fn = lambda args=prefix_args, kwargs=prefix_kwargs, fn=original_native: fn(
                            *args, **kwargs
                        )
                        prefix_metadata = {
                            "symbol": "aiter.mha_batch_prefill_func",
                            "output_dtype": str(tail_parameters["PART_OUT"].dtype),
                            "LSE_dtype": str(tail_parameters["PART_LSE"].dtype),
                            "LSE_layout": list(tail_parameters["PART_LSE"].shape),
                            "LSE_units": "natural_log",
                        }
                    else:
                        _require(
                            not native_calls and len(shared_recorder.records) == 1,
                            "unexpected Triton prefix dispatch",
                        )
                        prefix_fn = fixed_call(shared_recorder.records[0])
                        prefix_metadata = {
                            **shared_recorder.describe(),
                            "replaced_arguments": [],
                            "LSE_units": "log2",
                        }
                    tail_fn = fixed_call(tail_recorder.records[0])
                    # Whole capture always calls the complete original run.
                    # It never concatenates the independent stage closures,
                    # whose native LSE allocation identities can differ.
                    whole_fn = lambda ab=ab, case=case: ab.on_op.run(case.q, case.kv)
                    functions = {
                        "whole": whole_fn,
                        "prefix": prefix_fn,
                        "tail": tail_fn,
                    }
                    report32 = _comparison_report(
                        actual, fp32, atol=ATOL, rtol=RTOL, l2_limit=RELATIVE_L2_LIMIT
                    )
                    report64 = _comparison_report(
                        actual.double(),
                        fp64,
                        atol=ATOL,
                        rtol=RTOL,
                        l2_limit=RELATIVE_L2_LIMIT,
                    )
                    native_contract = (
                        ab.native_output_record() if native_route else None
                    )
                    route_valid = bool(tail_parameters["NATIVE_PREFIX"]) == native_route
                    route_valid = (
                        route_valid
                        and tail_parameters["PART_OUT"].dtype == torch.float32
                    )
                    expected_warps = (
                        1
                        if name
                        in ("native_FP32_layout_w1", "native_FP32_layout_w1_cut704")
                        else 2
                    )
                    route_valid = (
                        route_valid and tail_parameters["num_warps"] == expected_warps
                    )
                    if partial is not None:
                        route_valid = route_valid and bool(
                            (ab.on_op.cuts == arm_cuts[name]).all()
                        )
                        route_valid = route_valid and bool(
                            (ab.on_op.row_cuts == arm_cuts[name]).all()
                        )
                        route_valid = route_valid and bool(
                            (ab.on_op.cuts_host == arm_cuts[name]).all()
                        )
                        route_valid = route_valid and bool(
                            (ab.on_op.row_cuts_host == arm_cuts[name]).all()
                        )
                    route_valid = (
                        route_valid
                        and ab.on_op.BEAMS_PER_TILE == 128
                        and ab.on_op.splits == 1
                    )
                    if native_route:
                        route_valid = (
                            route_valid
                            and native_contract["native_output_contract_valid"]
                        )
                    else:
                        shared_parameters = shared_recorder.parameters()
                        route_valid = (
                            route_valid
                            and shared_parameters["PART_OUT"].dtype == torch.float32
                        )
                        route_valid = (
                            route_valid
                            and shared_parameters["PART_OUT"].data_ptr()
                            == tail_parameters["PART_OUT"].data_ptr()
                        )
                        route_valid = (
                            route_valid
                            and shared_parameters["PART_LSE"].data_ptr()
                            == tail_parameters["PART_LSE"].data_ptr()
                        )
                    single_graph_exact = {}
                    for scope, fn in functions.items():
                        graph, keepalive = capture(fn)
                        if scope in ("whole", "tail"):
                            single_graph_exact[scope] = torch.equal(
                                ab.on_op.output.contiguous().view(torch.uint8),
                                actual.contiguous().view(torch.uint8),
                            )
                        prepared[(name, scope)] = {
                            "single_graph": graph,
                            "single_keepalive": keepalive,
                            "fn": fn,
                            "tail_boundary_keepalive": tail_recorder,
                            "preflight_output": actual,
                        }
                    preflight[name] = {
                        "vs_fp32_all_rows": report32,
                        "vs_fp64_all_rows": report64,
                        "actual_route_valid": route_valid,
                        "single_graph_exact": single_graph_exact,
                        "native_output_contract": native_contract,
                    }
                    routes[name] = {
                        "prefix_route": (
                            "native_FP32_O"
                            if native_route
                            else "existing_Triton_FP32_partial_O"
                        ),
                        "prefix": prefix_metadata,
                        "tail": {**tail_recorder.describe(), "replaced_arguments": []},
                        "construction": {
                            "native_prefix_capable": ab.on_op.native_prefix,
                            "BEAMS_PER_TILE": ab.on_op.BEAMS_PER_TILE,
                            "splits": ab.on_op.splits,
                            "tiles": ab.on_op.tiles,
                        },
                        "test_native_prefix_byte_limit": limit,
                        **(
                            {
                                "manual_fixture_cut": arm_cuts[name],
                                "production_lineage": "NOT_IMPLEMENTED",
                            }
                            if partial is not None
                            else {}
                        ),
                    }
                del actual, native_calls, shared_recorder, tail_recorder
            workspace_addresses = {
                name: {
                    "output": ab.on_op.output.data_ptr(),
                    "native_prefix_output": ab.on_op.prefix_output.data_ptr(),
                    "triton_partial_output": ab.on_op.partial_out.data_ptr(),
                    "table": ab.on_op.table.data_ptr(),
                    "lengths": ab.on_op.lengths.data_ptr(),
                    "cuts": ab.on_op.cuts.data_ptr(),
                    "active_prefix_output": prepared[(name, "tail")][
                        "tail_boundary_keepalive"
                    ]
                    .parameters()["PART_OUT"]
                    .data_ptr(),
                    "active_prefix_LSE": prepared[(name, "tail")][
                        "tail_boundary_keepalive"
                    ]
                    .parameters()["PART_LSE"]
                    .data_ptr(),
                }
                for name, ab in operators.items()
            }
            independent = all(
                len({workspace_addresses[name][field] for name in names}) == len(names)
                for field in workspace_addresses[names[0]]
            )
            native_modules = _loaded_native_modules()
            native_valid = (
                len(native_modules) == 1
                and native_modules[0]["sha256"] == expected_native
            )
            emit(
                "real_performance_preflight_" + fixture.name,
                {
                    **case.describe(),
                    "classification": "real_attention_performance_preflight",
                    "arms": preflight,
                    "routes": routes,
                    "oracle_fp32_vs_fp64": oracle_report,
                    "independent_workspaces": independent,
                    "workspace_addresses": workspace_addresses,
                    "shared_input_addresses": {
                        "q": case.q.data_ptr(),
                        "kv": case.kv.data_ptr(),
                    },
                    "captured_GPU_input_hashes": input_before,
                    "loaded_native_modules": native_modules,
                    "expected_native_sha256": expected_native,
                    "timer_has_started": False,
                    **(
                        {"partial_prefix_fixture_lineage": lineage}
                        if partial is not None
                        else {}
                    ),
                },
            )
            self.assertTrue(oracle_report["thresholds_pass"])
            self.assertTrue(independent)
            self.assertTrue(native_valid)
            for record in preflight.values():
                self.assertTrue(record["vs_fp32_all_rows"]["thresholds_pass"])
                self.assertTrue(record["vs_fp64_all_rows"]["thresholds_pass"])
                self.assertTrue(record["actual_route_valid"])
                self.assertTrue(all(record["single_graph_exact"].values()))

            # Build each hot graph once, outside every timer and under a single
            # preparation patch. Every whole iteration is the original run.
            hot_graph_exact = {}
            for name in names:
                limit = original_limit if name != "triton_FP32_layout" else 0
                with mock.patch.object(
                    owner, "NATIVE_PREFIX_BYTE_LIMIT", limit
                ), mock.patch.object(cascade_mod, "_private_tail_merge", kernels[name]):
                    for scope in ("whole", "prefix", "tail"):
                        item = prepared[(name, scope)]
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            for _ in range(hot_unroll):
                                keepalive = item["fn"]()
                        graph.replay()
                        item["hot_graph"], item["hot_keepalive"] = graph, keepalive
                        if scope in ("whole", "tail"):
                            hot_graph_exact[name + "/" + scope] = torch.equal(
                                operators[name]
                                .on_op.output.contiguous()
                                .view(torch.uint8),
                                item["preflight_output"].contiguous().view(torch.uint8),
                            )
            torch.cuda.synchronize()
            emit(
                "real_performance_graphs_ready_" + fixture.name,
                {
                    "classification": "real_attention_performance_preflight",
                    "fixture": fixture.name,
                    "hot_graph_output_exact": hot_graph_exact,
                    "hot_unroll": hot_unroll,
                    "hot_samples_per_round": hot_samples,
                    "cold_driver_clear_bytes": cache_clear_bytes,
                    "timer_has_started": False,
                },
            )
            self.assertTrue(all(hot_graph_exact.values()))
            self.assertEqual(
                source_before, source_hashes(), "sources changed before timing"
            )
            self.assertEqual(owner.NATIVE_PREFIX_BYTE_LIMIT, original_limit)
            self.assertIs(cascade_mod._private_tail_merge, PRODUCTION_TAIL_KERNEL)
            arm_orders = [names[index:] + names[:index] for index in range(len(names))]
            arm_orders += [tuple(reversed(order)) for order in arm_orders]
            samples = {
                (name, scope): {"cold_graph_ms": [], "hot_unrolled_graph_ms": []}
                for name in names
                for scope in ("whole", "prefix", "tail")
            }
            orders = []
            for round_index, arm_order in enumerate(arm_orders):
                scopes = (
                    ("whole", "prefix", "tail")
                    if round_index < 3
                    else ("tail", "prefix", "whole")
                )
                order = [(name, scope) for scope in scopes for name in arm_order]
                modes = (
                    ("cold_graph_ms", "hot_unrolled_graph_ms")
                    if round_index % 2 == 0
                    else ("hot_unrolled_graph_ms", "cold_graph_ms")
                )
                orders.append(
                    {"arms": arm_order, "scopes": scopes, "cache_modes": modes}
                )
                for key in order:
                    item = prepared[key]
                    for mode in modes:
                        if mode == "cold_graph_ms":
                            measured = triton.testing.do_bench(
                                item["single_graph"].replay,
                                warmup=20,
                                rep=50,
                                return_mode="all",
                            )
                        else:
                            measured = hot_measure(item["hot_graph"])
                        samples[key][mode].append(list(measured))
            torch.cuda.synchronize()
            input_after, source_after = input_hashes(), source_hashes()
            native_after = _loaded_native_modules()
            summary = {}
            for (name, scope), modes in samples.items():
                key = name + "/" + scope
                summary[key] = {}
                for mode, rounds in modes.items():
                    values = sorted(value * 1000 for batch in rounds for value in batch)
                    medians = [statistics.median(batch) * 1000 for batch in rounds]
                    summary[key][mode] = {
                        "median_us": statistics.median(values),
                        "p90_nearest_rank_us": values[(9 * len(values) + 9) // 10 - 1],
                        "min_us": values[0],
                        "max_us": values[-1],
                        "sample_count": len(values),
                        "round_median_us": medians,
                        "round_median_max_over_min": max(medians) / min(medians),
                    }
            ratios = []
            comparisons = (
                ((names[0], names[1]), (names[1], names[2]), (names[0], names[2]))
                if comparison_mode in ("waves", "partial_prefix")
                else ((names[0], names[1]), (names[2], names[1]))
            )
            for baseline, compared in comparisons:
                for scope in ("whole", "prefix", "tail"):
                    for mode in ("cold_graph_ms", "hot_unrolled_graph_ms"):
                        left, right = (
                            summary[baseline + "/" + scope][mode],
                            summary[compared + "/" + scope][mode],
                        )
                        ratios.append(
                            {
                                "baseline": baseline,
                                "candidate": compared,
                                "scope": scope,
                                "cache_mode": mode,
                                "baseline_over_candidate": left["median_us"]
                                / right["median_us"],
                                "paired_round_ratios": [
                                    a / b
                                    for a, b in zip(
                                        left["round_median_us"],
                                        right["round_median_us"],
                                    )
                                ],
                            }
                        )
            emit(
                "real_attention_performance_" + fixture.name,
                {
                    **case.describe(),
                    "classification": "real_attention_kernel_performance_not_service_QPS",
                    "comparison_mode": comparison_mode,
                    "routes": routes,
                    "round_orders": orders,
                    "raw_samples": {
                        name + "/" + scope: modes
                        for (name, scope), modes in samples.items()
                    },
                    "summary": summary,
                    "baseline_over_candidate_ratios": ratios,
                    "source_sha256": source_before,
                    "sources_unchanged": source_before == source_after,
                    "input_hashes_unchanged": input_before == input_after,
                    "loaded_native_modules": native_modules,
                    "native_modules_unchanged": native_modules == native_after,
                    "expected_native_sha256": expected_native,
                    "workspace_addresses": workspace_addresses,
                    "independent_workspaces": independent,
                    "hot_graph_output_exact": hot_graph_exact,
                    "device": torch.cuda.get_device_name(),
                    "torch": str(torch.__version__),
                    "hip": torch.version.hip,
                    "triton": str(triton.__version__),
                    "timer_source_sha256": _sha_file(Path(triton.testing.__file__)),
                    "build_environment": {
                        name: os.getenv(name)
                        for name in (
                            "MKT_CASCADE_NATIVE_FP32_O",
                            "AITER_JIT_DIR",
                            "CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT",
                            "GPU_ARCHS",
                            "CU_NUM",
                            "AITER_AOT_IMPORT",
                            "MKT_CASCADE_EXPECTED_NATIVE_SHA256",
                        )
                    },
                    "timer": "GPU events; fixed single graphs and fixed 64-call unrolled graphs",
                    "cold_cache": {
                        "mode": "triton.testing.do_bench return_mode=all",
                        "warmup_ms": 20,
                        "rep_ms": 50,
                        "driver_clear_bytes": cache_clear_bytes,
                        "clear_outside_timed_events": True,
                        "scope": "best-effort eviction before each single replay",
                    },
                    "hot_cache": {
                        "unroll": hot_unroll,
                        "samples_per_round": hot_samples,
                        "untimed_warmup_replays_per_round": 2,
                        "rounds": len(arm_orders),
                        "scope": "same inputs repeatedly replayed; lower-bound latency",
                        "sample_semantics": "GPU batch elapsed time divided by 64; reported hot quantiles describe batch means, not individual request RT",
                    },
                    "whole_scope": "Original complete run connects each prefix invocation to its own returned LSE and tail; RoPE/cache insertion/CPU planning excluded",
                    "stage_scope": "Independent prefix-only and tail-only timings; do not add them. Native prefix-only includes its original API output-LSE allocation/capture scope; tail-only holds the recorded prefix O/LSE alive for fixed inputs.",
                    "constructor_scope": "All arms retain native-capable 128-beam/splits=1 construction; when selected, the Triton route uses a test-scoped byte-limit override",
                    "end_to_end_model_latency_and_QPS": "NOT_RUN",
                    **(
                        {
                            "partial_prefix_fixture_lineage": lineage,
                            "partial_prefix_planning_scope": "Manual cut prepared before timing; scheduler lineage, CPU planning and H2D excluded",
                        }
                        if partial is not None
                        else {}
                    ),
                },
            )
            self.assertEqual(source_before, source_after)
            self.assertEqual(input_before, input_after)
            self.assertEqual(native_modules, native_after)
            if partial is not None:
                cut_contexts.close()
            del prepared, operators, case, fp32, fp64
            del (
                functions,
                ab,
                item,
                graph,
                keepalive,
                whole_fn,
                prefix_fn,
                tail_fn,
                tail_parameters,
                shared_parameters,
            )

    def test_graph_real_layers_and_equivalent_metadata(self):
        groups = {}
        for fixture in self.fixtures:
            groups.setdefault(fixture.graph_key(), []).append(fixture)
        arena = self._arena()
        for group_index, fixtures in enumerate(groups.values()):
            anchor = max(
                fixtures,
                key=lambda f: f.manifest["paged_kv_metadata"]["storage_offset"],
            )
            case = RealCase(anchor, arena)
            ab = RealAttentionAB(case)
            off_graph, off = capture(ab.off)
            on_graph, on = capture(ab.on)

            def addresses():
                return (
                    case.q.data_ptr(),
                    case.kv.data_ptr(),
                    off.data_ptr(),
                    on.data_ptr(),
                    ab.params.seq_lens.data_ptr(),
                    ab.params.kv_cache_block_id_device.data_ptr(),
                    ab.on_op.row_cuts.data_ptr(),
                    ab.on_op.table.data_ptr(),
                )

            original_addresses = addresses()
            steps = [(fixture, False) for fixture in fixtures]
            steps.extend(((fixtures[-1], True), (fixtures[0], False)))
            previous_kv_hash = anchor.manifest["tensors"]["kv_pages"]["sha256"]
            for step, (fixture, permute) in enumerate(steps):
                previous_q, previous_table = case.q.clone(), case.table.clone()
                new = RealCase(fixture, arena, permute=permute)
                q_changed = not torch.equal(previous_q, new.q)
                table_changed = not torch.equal(previous_table, new.table)
                if permute:
                    self.assertTrue(
                        table_changed, "equivalent remapping did not change metadata"
                    )
                ab.update(new)
                off.fill_(float("nan"))
                on.fill_(float("nan"))
                off_graph.replay()
                on_graph.replay()
                replay_off, replay_on = off.clone(), on.clone()
                eager_off, eager_on = ab.off().clone(), ab.on().clone()
                torch.testing.assert_close(replay_off, eager_off, atol=0, rtol=0)
                torch.testing.assert_close(replay_on, eager_on, atol=0, rtol=0)
                self.assertEqual(addresses(), original_addresses)
                label = f"real_graph_group{group_index}_step{step}_{fixture.name}"
                restored = self._captured_off_check(case, replay_off, label)
                metrics = self._accuracy(ab, replay_off, replay_on)
                kv_hash = fixture.manifest["tensors"]["kv_pages"]["sha256"]
                emit(
                    label,
                    {
                        **case.describe(),
                        "cascade_prefix": ab.prefix_path(),
                        "graph_group_count": len(groups),
                        "compatible_real_layers": [
                            f.manifest["layer_idx"] for f in fixtures
                        ],
                        "graph_anchor_layer": anchor.manifest["layer_idx"],
                        "stable_addresses": True,
                        "q_values_changed": q_changed,
                        "kv_values_changed": kv_hash != previous_kv_hash,
                        "physical_table_changed": table_changed,
                        "history_update_scope": "copied actual captured history; no fabricated lengths",
                        "metadata_variant": case.variant,
                        "off_vs_captured": restored,
                        "metrics": metrics,
                        "graph_vs_own_eager": {
                            "off": error_metrics(replay_off, eager_off),
                            "on": error_metrics(replay_on, eager_on),
                        },
                        "graph_vs_own_eager_thresholds": {"atol": 0, "rtol": 0},
                        "end_to_end_model_accuracy": "NOT_RUN",
                    },
                )
                previous_kv_hash = kv_hash
                del (
                    new,
                    previous_q,
                    previous_table,
                    replay_off,
                    replay_on,
                    eager_off,
                    eager_on,
                )
            del ab, case, off, on, off_graph, on_graph


if __name__ == "__main__":
    import unittest

    unittest.main()
