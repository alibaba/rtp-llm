"""Compare raw gRPC outputs and require execution records proving co-batching."""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import grpc
import numpy as np
from tokenizers import Tokenizer

from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub

CONTINUATION = " red blue green yellow"
TARGET_LENGTH = 193
SHORT_LENGTH = 11
LONG_LENGTH = 321
BLOCKER_STEPS = 64


def read_vocab_size(checkpoint):
    with (checkpoint / "config.json").open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    vocab_size = config.get("vocab_size")
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


def start_request(stub, ids, steps, *, measure=False):
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
        return_logprobs=measure,
    )
    generate.aux_info.value = measure
    if measure:
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
    tokens, logits, softmax_probs, token_logprobs = [], [], [], []
    for response in call:
        output = response.flatten_output
        ids = read_tensor(output.output_ids, pb.TensorPB.INT32, "int32_data", "<i4")
        if not ids.size:
            continue
        row = read_tensor(output.logits, pb.TensorPB.FP32, "fp32_data", "<f4")
        probs = read_tensor(
            output.aux_info[0].softmax_probs, pb.TensorPB.FP32, "fp32_data", "<f4"
        )
        logprob = read_tensor(
            output.token_logprobs, pb.TensorPB.FP32, "fp32_data", "<f4"
        )
        assert list(output.logprobs_offsets) == [0]
        assert list(output.logprobs_counts) == [1]
        assert ids.size == probs.size == logprob.size == 1
        assert row.ndim and row.shape[-1] == vocab_size and row.size == vocab_size, (
            row.shape,
            vocab_size,
        )
        assert np.isfinite(row).all() and 0 < probs.item() <= 1
        assert np.isfinite(logprob.item())
        tokens.append(ids.item())
        logits.append(row.reshape(vocab_size))
        softmax_probs.append(probs.item())
        token_logprobs.append(logprob.item())
    assert len(tokens) == steps, (len(tokens), steps, tokens)
    return {
        "tokens": np.asarray(tokens, dtype=np.int32),
        "logits": np.stack(logits),
        "softmax_probs": np.asarray(softmax_probs, dtype=np.float32),
        "token_logprobs": np.asarray(token_logprobs, dtype=np.float32),
    }


def compare(reference, actual, name):
    differences, first_differences = {}, {}
    for field, second in actual.items():
        first = reference[field]
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
        "steps": len(actual["tokens"]),
        "logits_shape": list(actual["logits"].shape),
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


def real_rows(batch):
    return [row for row in batch["requests"] if not row.get("is_fake")]


def verify_execution(log_dir, offsets, neighbors=(), target_first=True, chunked=False):
    # Require the full prefill history so an async log flush from the previous
    # request cannot satisfy the check.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        targets = defaultdict(list)
        for batch in read_records(log_dir, offsets):
            rows = real_rows(batch)
            for i, row in enumerate(rows):
                if row["input_len"] == TARGET_LENGTH:
                    targets[row["request_id"]].append((i, rows))

        for records in targets.values():
            prefill = []
            mixed_phases = set()
            for i, rows in records:
                row = rows[i]
                phase = row["phase"]
                if phase == "prefill":
                    prefill.append((row["kv_len"], row["q_len"]))
                wanted_row = 0 if target_first else len(rows) - 1
                lengths = {peer["input_len"] for peer in rows if peer["phase"] == phase}
                if neighbors and i == wanted_row and set(neighbors) <= lengths:
                    mixed_phases.add(phase)
            grants = [size for _, size in sorted(prefill)]
            expected = [193]
            if chunked:
                expected = [192, 1] if target_first else [128, 65]
            if grants != expected:
                continue
            if neighbors:
                if mixed_phases != {"prefill", "decode"}:
                    continue
            elif not all(len(rows) == 1 for _, rows in records):
                continue
            print(
                f"target prefill grants {grants}, mixed phases {sorted(mixed_phases)}"
            )
            return
        time.sleep(0.1)
    raise AssertionError(
        f"missing execution evidence for {neighbors=}, {target_first=}, {chunked=}"
    )


def run_mixed(stub, blocker, order, target, steps, vocab_size):
    # Keep the worker busy so all the measured requests can queue together.
    hold = start_request(stub, blocker, BLOCKER_STEPS)
    next(hold)
    calls = []
    for ids in order:
        calls.append(start_request(stub, ids, steps, measure=ids is target))
        time.sleep(0.025)
    for _ in hold:
        pass
    for ids, call in zip(order, calls):
        if ids is target:
            result = collect(call, steps, vocab_size)
        else:
            for _ in call:
                pass
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--address", default="127.0.0.1:19481")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--save", type=Path, help="Save the unchunked solo reference")
    budget.add_argument(
        "--reference", type=Path, help="Compare budget 192 to this file"
    )
    args = parser.parse_args()
    chunked = args.reference is not None
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    vocab_size = read_vocab_size(args.checkpoint)
    fixed = tokenizer.encode(CONTINUATION, add_special_tokens=False).ids
    target = make_prompt(tokenizer, TARGET_LENGTH, " The ocean is calm.")
    short = make_prompt(tokenizer, SHORT_LENGTH, " Good.")
    long = make_prompt(tokenizer, LONG_LENGTH, " An oak grew near the hill.")
    blocker = make_prompt(tokenizer, 37, " Write a long story.")
    with grpc.insecure_channel(args.address) as channel:
        grpc.channel_ready_future(channel).result(timeout=600)
        stub = RpcServiceStub(channel)
        offsets = record_offsets(args.log_dir)
        solo = collect(
            start_request(stub, target, len(fixed), measure=True),
            len(fixed),
            vocab_size,
        )
        verify_execution(args.log_dir, offsets, chunked=chunked)
        np.testing.assert_array_equal(solo["tokens"], fixed)
        baseline = solo
        if chunked:
            with np.load(args.reference) as saved:
                if not solo.keys() <= set(saved.files):
                    parser.error("Regenerate the unchunked reference with --save")
                baseline = {field: saved[field] for field in solo}
            compare(baseline, solo, "chunked solo")
            cases = ((target, short), (short, target))
            neighbors = (SHORT_LENGTH,)
        else:
            np.savez(args.save, **solo)
            cases = ((target, short, long), (long, short, target))
            neighbors = (SHORT_LENGTH, LONG_LENGTH)
        for order in cases:
            target_first = order[0] is target
            offsets = record_offsets(args.log_dir)
            result = run_mixed(stub, blocker, order, target, len(fixed), vocab_size)
            verify_execution(args.log_dir, offsets, neighbors, target_first, chunked)
            compare(baseline, result, f"target {'first' if target_first else 'last'}")


if __name__ == "__main__":
    main()
