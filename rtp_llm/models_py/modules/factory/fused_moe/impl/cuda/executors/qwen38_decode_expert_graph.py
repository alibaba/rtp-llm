"""Rank-local CUDA Graphs for the World32 BF16 reference MoE decode path."""

from collections import Counter
import json
import os
import time
from pathlib import Path

import torch

RUNS = {}
INPUT_NAMES = ("hidden_states", "topk_ids", "topk_weights")


def _input_signature(inputs):
    return tuple(
        (tuple(x.shape), tuple(x.stride()), str(x.dtype), str(x.device), x.is_contiguous())
        for x in inputs
    )


def _describe_signature(signature):
    return {
        name: dict(shape=shape, stride=stride, dtype=dtype, device=device, contiguous=contiguous)
        for name, (shape, stride, dtype, device, contiguous) in zip(INPUT_NAMES, signature)
    }


def _cuda_device(device):
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("Qwen3.8 expert CUDA Graph requires a CUDA device")
    return device if device.index is not None else torch.device("cuda", torch.cuda.current_device())


class ExpertGraphRun:
    def __init__(self, rank, device, num_layers, receipt_dir, incarnation):
        self.rank = rank
        self.device = _cuda_device(device)
        self.num_layers = num_layers
        self.incarnation = incarnation
        self.refit_epoch = 0
        self.refitting = False
        self.decode_forwards = 0
        self.capture_stream = None
        self.graphs = []
        self.fallback_attempts = Counter()
        self.receipt = Path(receipt_dir) / f"q38-expert-graph-rank-{rank:05d}.jsonl"
        with self.receipt.open("x"):
            pass
        self.emit("configured", layers=num_layers, captured_rows=32, eager_fallback=True)

    def emit(self, event, **fields):
        record = {
            "scope": "qwen38_bf16_ep32_decode_local_routed_experts",
            "event": event,
            "incarnation": self.incarnation,
            "world_rank": self.rank,
            "pid": os.getpid(),
            "device": str(self.device),
            "refit_epoch": self.refit_epoch,
            "unix_ns": time.time_ns(),
            **fields,
        }
        with self.receipt.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def emit_progress(self, event):
        self.emit(
            event,
            decode_forwards=self.decode_forwards,
            captured_layers=sum(layer.graph is not None for layer in self.graphs),
            replay_launches=sum(layer.epoch_replays for layer in self.graphs),
            layers_verified=sum(layer.verified_epoch == self.refit_epoch for layer in self.graphs),
            eager_fallback_launches=sum(layer.epoch_fallbacks for layer in self.graphs),
            fallback_attempts_by_input=[
                {"inputs": _describe_signature(signature), "attempts": attempts}
                for signature, attempts in self.fallback_attempts.items()
            ],
        )

    def note_eager_fallback(self, layer, inputs):
        signature = _input_signature(inputs)
        self.fallback_attempts[signature] += 1
        if self.fallback_attempts[signature] == 1:
            self.emit("decode_eager_fallback", layer=layer, inputs=_describe_signature(signature))

    def begin_refit(self):
        if self.refitting:
            raise RuntimeError("Qwen3.8 expert graph refit already active")
        if self.refit_epoch:
            self.emit_progress("epoch_drained")
        self.refitting = True

    def finish_refit(self, counts):
        if not self.refitting:
            raise RuntimeError("Qwen3.8 expert graph received an unmatched refit completion")
        self.refit_epoch += 1
        self.decode_forwards = 0
        self.fallback_attempts.clear()
        self.refitting = False
        for layer in self.graphs:
            layer.epoch_replays = 0
            layer.epoch_fallbacks = 0
        self.emit("local_refit_complete", **counts)

    def note_last_layer(self):
        self.decode_forwards += 1
        count = self.decode_forwards
        if (count & (count - 1)) == 0 or count % 1024 == 0:
            self.emit_progress("epoch_progress")


def register_run(rank, device, num_layers):
    device = _cuda_device(device)
    if device in RUNS:
        raise RuntimeError("Qwen3.8 expert graph requires one rank-local policy per device")
    directory = Path(os.environ["RTP_WEIGHT_MANIFEST_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    run = ExpertGraphRun(rank, device, num_layers, directory, os.environ["RTP_LLM_RL_RUN_ID"])
    RUNS[device] = run
    return run


def begin_refit(device):
    RUNS[_cuda_device(device)].begin_refit()


def finish_refit(device, counts):
    RUNS[_cuda_device(device)].finish_refit(counts)


class DecodeExpertGraph:
    def __init__(self, run, layer, w1, w2):
        expected = ((16, 4096, 8192), (16, 8192, 2048))
        if any(w.shape != shape or w.device != run.device or w.dtype != torch.bfloat16
               for w, shape in zip((w1, w2), expected)):
            raise RuntimeError("Qwen3.8 expert graph requires the exact unquantized EP32 weights")
        self.run = run
        self.layer = layer
        self.weight_ptrs = (w1.data_ptr(), w2.data_ptr())
        self.graph = None
        self.stream = None
        self.inputs = None
        self.output = None
        self.verified_epoch = 0
        self.epoch_replays = 0
        self.epoch_fallbacks = 0
        self.failed = False
        run.graphs.append(self)

    def _stage_inputs(self, inputs):
        for dest, src in zip(self.inputs, inputs):
            dest.copy_(src)

    def _capture(self, compute, inputs, stream):
        self.inputs = tuple(torch.empty_like(x) for x in inputs)
        self._stage_inputs(inputs)
        self.stream = stream
        if self.run.capture_stream is None:
            self.run.capture_stream = torch.cuda.Stream(device=self.run.device)
        capture_stream = self.run.capture_stream
        capture_stream.wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(capture_stream):
            reference = compute(*self.inputs)
            graph.capture_begin(capture_error_mode="thread_local")
            try:
                output = compute(*self.inputs)
            except BaseException:
                try:
                    graph.capture_end()
                except RuntimeError:
                    pass
                raise
            graph.capture_end()
        stream.wait_stream(capture_stream)
        reference.record_stream(stream)
        self.output = output
        self.graph = graph
        return reference

    def execute(self, compute, inputs, w1, w2):
        if self.failed or self.run.refitting or not self.run.refit_epoch:
            raise RuntimeError("Qwen3.8 expert graph requires a successful local refit and healthy capture")
        if self.weight_ptrs != (w1.data_ptr(), w2.data_ptr()):
            raise RuntimeError("Qwen3.8 expert graph persistent weight addresses changed")
        specs = (((32, 8192), torch.bfloat16), ((32, 10), torch.int32), ((32, 10), torch.float32))
        if all(x.shape == shape and x.dtype == dtype and x.device == self.run.device and x.is_contiguous()
               for x, (shape, dtype) in zip(inputs, specs)):
            output = self._replay(compute, inputs)
            self.epoch_replays += 1
        else:
            # Other attention-DP ranks can supply prefill rows to this eager EP
            # gather even when this rank is decoding. Keep their original layout.
            self.run.note_eager_fallback(self.layer, inputs)
            output = compute(*inputs)
            self.epoch_fallbacks += 1

        if self.layer == self.run.num_layers - 1:
            self.run.note_last_layer()
        return output

    def _replay(self, compute, inputs):
        stream = torch.cuda.current_stream(self.run.device)
        if self.stream is not None and self.stream != stream:
            raise RuntimeError("Qwen3.8 expert graph decode stream changed")

        needs_capture = self.graph is None
        needs_check = self.verified_epoch != self.run.refit_epoch
        reference = None
        try:
            if needs_capture:
                started = time.monotonic_ns()
                reference = self._capture(compute, inputs, stream)
            else:
                self._stage_inputs(inputs)
                if needs_check:
                    started = time.monotonic_ns()
                    reference = compute(*inputs)

            self.graph.replay()
            if needs_check:
                if not torch.equal(self.output.view(torch.int16), reference.view(torch.int16)):
                    raise RuntimeError("Qwen3.8 expert graph replay differs from eager expert output bits")
                self.verified_epoch = self.run.refit_epoch
                self.run.emit(
                    "layer_replay_verified",
                    layer=self.layer,
                    captured=needs_capture,
                    validation_wall_ns=time.monotonic_ns() - started,
                )
        except BaseException as exc:
            self.failed = True
            self.run.emit("failure", layer=self.layer, error=repr(exc))
            raise

        return self.output
