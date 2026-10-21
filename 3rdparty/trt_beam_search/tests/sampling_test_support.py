"""Test-only exact CUDA oracle and explicit Sampling inputs (no service changes)."""
import hashlib
import json
import os
from pathlib import Path

import torch

def data_file(relative):
    # PYTHONPATH may import this module from the source tree even under bazel
    # test. Generated artifacts must be resolved from the test's runfiles.
    candidates = []
    for variable in ("RUNFILES_DIR", "TEST_SRCDIR"):
        root = os.environ.get(variable)
        if root:
            for workspace in dict.fromkeys((os.environ.get("TEST_WORKSPACE", "rtp_llm"), "rtp_llm")):
                candidates.append(Path(root) / workspace / "3rdparty/trt_beam_search/tests" / relative)
    candidates.append(Path(__file__).absolute().parent / relative)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing test data {relative}; checked: {candidates}")


def load_ops():
    torch.ops.load_library(str(data_file("beam_search_test_ops.so")))
    return torch.ops.rtp_beam_sampling_test


def check_frozen_sources():
    manifest = json.loads(data_file("frozen/manifest.json").read_text())
    for item in manifest["files"]:
        assert hashlib.sha256(data_file("frozen/" + item["file"]).read_bytes()).hexdigest() == item["sha256"]
    return manifest


def exact(left, right):
    if isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for i, (a, b) in enumerate(zip(left, right)):
            try:
                exact(a, b)
            except AssertionError as exc:
                raise AssertionError(f"output field {i}: {exc}") from exc
        return
    assert left.shape == right.shape and left.dtype == right.dtype
    a = left.detach().contiguous().view(torch.uint8).cpu()
    b = right.detach().contiguous().view(torch.uint8).cpu()
    assert torch.equal(a, b), f"byte mismatch: {left.shape} {left.dtype} ({int((a != b).sum())} bytes)"


def clones(values):
    return [x.clone() for x in values]


def empty(dtype):
    return torch.empty(0, dtype=dtype, device="cuda")


def add_inputs(batch, beams, topk, dtype=torch.float32, tail=0, states=False, seed=123):
    torch.manual_seed(seed)
    size = beams * topk - tail
    values = torch.randn(batch, size, device="cuda", dtype=dtype)
    cum = torch.randn(batch + 1 if states else batch, beams, device="cuda", dtype=torch.float32)
    finished, ends, diversity, slots = empty(torch.uint8), empty(torch.int32), empty(torch.float32), empty(torch.int32)
    if states:
        finished = torch.zeros_like(cum, dtype=torch.uint8)
        finished[:, ::3] = 1
        ends = torch.arange(cum.shape[0], device="cuda", dtype=torch.int32)
        diversity = torch.linspace(-.25, .125, cum.shape[0], device="cuda", dtype=torch.float32)
        slots = torch.arange(batch, 0, -1, device="cuda", dtype=torch.int32)
    return [values, cum, finished, ends, diversity, slots]


def sample_inputs(batch, beams, vocab=24576, seed=123, pattern="random"):
    torch.manual_seed(seed)
    logits = torch.randn(batch, beams, vocab, device="cuda", dtype=torch.float32)
    if pattern == "ties":
        logits.zero_()
    elif pattern == "near":
        logits = (torch.arange(vocab, device="cuda") % 7).float().expand(batch, beams, -1).clone() * 1e-7
    elif pattern == "masked":
        logits[..., ::3] = -torch.inf
        logits[..., 1] = 80.0
        logits[..., 2] = -80.0
    ids = (torch.arange(batch * beams * 503, device="cuda", dtype=torch.int32) % vocab).reshape(batch, beams, 503)
    inputs = torch.full((batch, beams), 500, device="cuda", dtype=torch.int32)
    lengths = torch.full_like(inputs, 500 if beams == 1 else 501)
    cum = torch.randn(batch, beams, device="cuda", dtype=torch.float32)
    if pattern == "ties":
        cum.zero_()
    return [logits, ids, inputs, lengths, cum]


def stage_inputs(batch, beams, output, dtype=torch.float32, states=False, v2=True, seed=123):
    torch.manual_seed(seed)
    mbs = batch + 1 if states else batch
    vocab = 257
    count = 2 * output if v2 else beams * output * 2
    # Already sorted Stage 2 values. Parent IDs intentionally repeat and vary.
    order = torch.arange(count, device="cuda", dtype=torch.int32)
    ids = ((order * 7 % beams) * vocab + (order * 11 + 1) % vocab).expand(batch, -1).clone()
    scores = (-torch.arange(count, device="cuda").float() * .0078125).to(dtype).expand(batch, -1).clone()
    lengths = (torch.arange(mbs * beams, device="cuda", dtype=torch.int32) % 3 + 3).reshape(mbs, beams)
    inputs = torch.ones_like(lengths)
    cum = torch.randn(mbs, beams, device="cuda", dtype=torch.float32)
    finished, ends, slots = empty(torch.uint8), empty(torch.int32), empty(torch.int32)
    if states:
        finished = torch.zeros(mbs, output, device="cuda", dtype=torch.uint8)
        finished[:, ::3] = 1
        ends = torch.ones(mbs, device="cuda", dtype=torch.int32)
        slots = torch.arange(batch, 0, -1, device="cuda", dtype=torch.int32)
    return [ids, scores, lengths, inputs, cum, finished, ends, slots]


def stage_buffers(state, output):
    """Allocate low-level scratch outside benchmark measurement/capture."""
    batch, mbs, beams = state[0].shape[0], *state[2].shape
    def integer(shape, value=0):
        return torch.full(shape, value, dtype=torch.int32, device="cuda")
    def floating(shape, value=0.):
        return torch.full(shape, value, dtype=torch.float32, device="cuda")
    return [integer((mbs, output)), integer((mbs, output)),
            integer((mbs, output), -123), integer((mbs, output), -123), floating((mbs, output), -123),
            floating((12, mbs, output), -123), integer((mbs,)), floating((mbs,)),
            integer((mbs, output * 2, 12), -123), integer((mbs, output * 2), -123),
            floating((mbs, output * 2), -123), floating((mbs, output * 2), -123),
            torch.zeros(mbs, dtype=torch.bool, device="cuda"),
            torch.empty(batch * beams * output * 2, dtype=torch.float32, device="cuda")]
