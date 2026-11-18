"""Compare raw gRPC outputs and require execution records proving co-batching."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import grpc
import numpy as np
from tokenizers import Tokenizer
import torch

from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub

CONTINUATION = " red blue green yellow"
TARGET_LENGTH = 193
SHORT_LENGTH = 11
LONG_LENGTH = 321


@dataclass
class Output:
    tokens: np.ndarray
    logits: np.ndarray
    logprobs: np.ndarray
    sampled_logprobs: np.ndarray


def read_vocab_size(checkpoint):
    with (checkpoint / "config.json").open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    text_config = config.get("text_config") or {}
    vocab_size = text_config.get("vocab_size", config.get("vocab_size"))
    assert isinstance(vocab_size, int) and vocab_size > 0, (checkpoint, vocab_size)
    return vocab_size


def make_prompt(tokenizer, length, lead):
    instruction = tokenizer.encode(
        "\nContinue with the text: red blue green yellow\n", add_special_tokens=False
    ).ids
    text = tokenizer.encode(lead * (length + 1), add_special_tokens=False).ids
    prompt = text[: length - len(instruction)] + instruction
    assert len(prompt) == length
    return prompt


def start_request(stub, ids, steps, *, measure=False, fixed=True):
    generate = pb.GenerateConfigPB(
        max_new_tokens=steps,
        num_beams=1,
        num_return_sequences=1,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
        repetition_penalty=1.0,
        ignore_eos=True,
        return_incremental=True,
        is_streaming=True,
        timeout_ms=180_000,
        return_logits=measure,
        return_softmax_probs=measure,
    )
    generate.aux_info.value = measure
    if measure and fixed:
        generate.regex.value = CONTINUATION
    request = pb.GenerateInputPB(
        request_id=time.time_ns(), token_ids=ids, generate_config=generate
    )
    return stub.GenerateStreamCall(request, timeout=180)


def read_tensor(tensor, data_type, field, dtype):
    assert tensor.data_type == data_type
    return np.frombuffer(getattr(tensor, field), dtype=dtype).reshape(
        tuple(tensor.shape)
    )


def collect(call, steps, vocab_size):
    tokens, logits, sampled = [], [], []
    for response in call:
        output = response.flatten_output
        ids = read_tensor(output.output_ids, pb.TensorPB.INT32, "int32_data", "<i4")
        if not ids.size:
            continue
        row = read_tensor(output.logits, pb.TensorPB.FP32, "fp32_data", "<f4")
        probs = read_tensor(
            output.aux_info[0].softmax_probs, pb.TensorPB.FP32, "fp32_data", "<f4"
        )
        assert ids.size == probs.size == 1, (ids.shape, probs.shape)
        assert row.ndim and row.shape[-1] == vocab_size and row.size == vocab_size, (
            row.shape,
            vocab_size,
        )
        assert np.isfinite(row).all() and 0 < probs.item() <= 1
        tokens.append(ids.item())
        logits.append(row.reshape(vocab_size))
        sampled.append(probs.item())
    assert len(tokens) == steps, (len(tokens), steps, tokens)
    logits = np.stack(logits)
    assert logits.shape == (steps, vocab_size), logits.shape
    logprobs = torch.log_softmax(torch.from_numpy(logits), dim=-1).numpy()
    sampled_logprobs = np.log(np.asarray(sampled, dtype=np.float32))
    return Output(
        np.asarray(tokens, dtype=np.int32), logits, logprobs, sampled_logprobs
    )


def compare(reference, actual, name):
    differences, first_differences = {}, {}
    for field in Output.__annotations__:
        first, second = getattr(reference, field), getattr(actual, field)
        assert first.dtype == second.dtype and first.shape == second.shape
        first_bits, second_bits = first.view(np.uint32), second.view(np.uint32)
        mismatch = first_bits != second_bits
        differences[field] = int(np.count_nonzero(mismatch))
        if differences[field]:
            index = np.unravel_index(mismatch.argmax(), mismatch.shape)
            first_differences[field] = {
                "index": [int(position) for position in index],
                "reference": first[index].item(),
                "actual": second[index].item(),
                "reference_bits": hex(first_bits[index]),
                "actual_bits": hex(second_bits[index]),
            }
    result = {
        "case": name,
        "steps": len(actual.tokens),
        "logits_shape": list(actual.logits.shape),
        "different_elements": differences,
        "first_differences": first_differences,
    }
    print(json.dumps(result), flush=True)
    assert not first_differences, f"{name}: {differences}"


def record_offsets(log_dir):
    return {path: path.stat().st_size for path in log_dir.glob("batch_schedule.*.log")}


def read_records(log_dir, offsets):
    for path in log_dir.glob("batch_schedule.*.log"):
        with path.open("r", encoding="utf-8") as file:
            file.seek(offsets.get(path, 0))
            for line in file:
                start = line.find("{")
                if start >= 0 and line.endswith("\n"):
                    yield json.loads(line[start:])


def evidence(log_dir, offsets, *, neighbors=(), target_first=True, chunked=False):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        target_prefill, shared_prefill, shared_decode = [], [], []
        target_was_alone = True
        for batch in read_records(log_dir, offsets):
            rows = [row for row in batch.get("requests", ()) if not row.get("is_fake")]
            for i, row in enumerate(rows):
                if row["input_len"] != TARGET_LENGTH:
                    continue
                target_was_alone &= len(rows) == 1
                phase = row["phase"]
                if phase == "prefill":
                    target_prefill.append((row.get("kv_len", 0), row.get("q_len", 0)))
                wanted_row = 0 if target_first else len(rows) - 1
                if not neighbors or i != wanted_row:
                    continue
                lengths = {peer["input_len"] for peer in rows if peer["phase"] == phase}
                if set(neighbors).issubset(lengths):
                    record = (batch["exec_id"], i, [peer["input_len"] for peer in rows])
                    if phase == "prefill":
                        shared_prefill.append(record)
                    elif phase == "decode":
                        shared_decode.append(record)
        complete_prefill = sum(n for _, n in target_prefill) == TARGET_LENGTH
        saw_cut = any(prefix > 0 for prefix, _ in target_prefill)
        if (
            complete_prefill
            and saw_cut == chunked
            and (
                (not neighbors and target_was_alone)
                or (shared_prefill and shared_decode)
            )
        ):
            print(
                f"target prefill (prefix, grant) {sorted(target_prefill)}", flush=True
            )
            for name, records in (
                ("prefill", shared_prefill),
                ("decode", shared_decode),
            ):
                if records:
                    print(
                        f"mixed {name} (exec id, target row, prompt lengths) {records}",
                        flush=True,
                    )
            return
        time.sleep(0.1)
    raise AssertionError(
        f"missing execution evidence for neighbors {neighbors}, "
        f"target_first={target_first}, chunked={chunked}: "
        f"{target_prefill=}, {shared_prefill=}, {shared_decode=}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--address", default="127.0.0.1:19481")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--save", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--chunked", action="store_true")
    smoke = parser.add_mutually_exclusive_group()
    smoke.add_argument("--greedy-smoke", "--off-smoke", action="store_true")
    smoke.add_argument("--greedy-logits", action="store_true")
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    vocab_size = read_vocab_size(args.checkpoint)
    fixed = tokenizer.encode(CONTINUATION, add_special_tokens=False).ids
    target = make_prompt(tokenizer, TARGET_LENGTH, " The ocean is calm.")
    short = make_prompt(tokenizer, SHORT_LENGTH, " Good.")
    long = make_prompt(tokenizer, LONG_LENGTH, " An oak grew near the hill.")
    blocker = make_prompt(tokenizer, 37, " Write a long story.")
    args.log_dir = args.log_dir.resolve()
    with grpc.insecure_channel(args.address) as channel:
        grpc.channel_ready_future(channel).result(timeout=600)
        stub = RpcServiceStub(channel)
        if args.greedy_logits:
            output = collect(
                start_request(stub, target, 8, measure=True, fixed=False), 8, vocab_size
            )
            if args.save:
                np.savez(args.save, **vars(output))
            print(
                json.dumps(
                    dict(
                        tokens=output.tokens.tolist(),
                        shape=output.logits.shape,
                        dtype=str(output.logits.dtype),
                        finite=output.logits.size,
                    )
                ),
                flush=True,
            )
            return
        if args.greedy_smoke:
            tokens = []
            for response in start_request(stub, target, 8):
                output = response.flatten_output.output_ids
                tokens.extend(
                    read_tensor(
                        output, pb.TensorPB.INT32, "int32_data", "<i4"
                    ).flatten()
                )
            assert len(tokens) == 8
            print(f"greedy tokens: {list(map(int, tokens))}", flush=True)
            return

        offsets = record_offsets(args.log_dir)
        solo = collect(
            start_request(stub, target, len(fixed), measure=True),
            len(fixed),
            vocab_size,
        )
        evidence(args.log_dir, offsets, chunked=args.chunked)
        baseline = solo
        if args.reference:
            with np.load(args.reference) as reference:
                baseline = Output(
                    **{field: reference[field] for field in Output.__annotations__}
                )
            compare(baseline, solo, "cross-budget solo")
        np.testing.assert_array_equal(solo.tokens, fixed)
        print(f"fixed continuation tokens: {solo.tokens.tolist()}", flush=True)
        if args.save:
            np.savez(args.save, **vars(solo))

        # With a 192-token budget the target's first grant fills the batch.
        # The remaining target token can share a round with the short request.
        cases = (
            (
                ((target, short), True, (SHORT_LENGTH,)),
                ((short, target), False, (SHORT_LENGTH,)),
            )
            if args.chunked
            else (
                ((target, short, long), True, (SHORT_LENGTH, LONG_LENGTH)),
                ((long, short, target), False, (SHORT_LENGTH, LONG_LENGTH)),
            )
        )
        for order, target_first, neighbors in cases:
            offsets = record_offsets(args.log_dir)
            # Queue the peers behind an active request so they reach the same
            # FIFO admission round. Execution records independently verify it.
            hold = start_request(stub, blocker, 64)
            next(hold)
            calls = []
            for ids in order:
                calls.append(
                    start_request(stub, ids, len(fixed), measure=ids is target)
                )
                time.sleep(0.025)
            for _ in hold:
                pass
            for ids, call in zip(order, calls):
                if ids is target:
                    actual = collect(call, len(fixed), vocab_size)
                else:
                    for _ in call:
                        pass
            evidence(
                args.log_dir,
                offsets,
                neighbors=neighbors,
                target_first=target_first,
                chunked=args.chunked,
            )
            compare(baseline, actual, f"target {'first' if target_first else 'last'}")


if __name__ == "__main__":
    main()
