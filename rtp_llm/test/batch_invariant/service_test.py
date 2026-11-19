"""Compare raw gRPC outputs and require execution records proving co-batching."""

import argparse
import json
import time
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import grpc
import numpy as np
import torch
from tokenizers import Tokenizer

from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub

CONTINUATION = " red blue green yellow"
TARGET_LENGTH = 193
SHORT_LENGTH = 11
LONG_LENGTH = 321
BLOCKER_STEPS = 64
PEER_STEPS = 128
COMPARED_FIELDS = ("tokens", "logits", "logprobs", "sampled_logprobs")


@dataclass
class Output:
    tokens: np.ndarray
    logits: np.ndarray
    logprobs: np.ndarray
    sampled_logprobs: np.ndarray
    sampled_probs: np.ndarray | None = None


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


def make_request(ids, steps, *, measure=False, fixed=True, streaming=True):
    generate = pb.GenerateConfigPB(
        max_new_tokens=steps,
        num_beams=1,
        num_return_sequences=1,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
        repetition_penalty=1.0,
        ignore_eos=True,
        return_incremental=streaming,
        is_streaming=streaming,
        timeout_ms=180_000,
        return_logits=measure,
        return_softmax_probs=measure,
    )
    generate.aux_info.value = measure
    if measure and fixed:
        generate.regex.value = CONTINUATION
    return pb.GenerateInputPB(
        request_id=time.time_ns(), token_ids=ids, generate_config=generate
    )


def start_request(stub, ids, steps, **kwargs):
    request = make_request(ids, steps, **kwargs)
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
    sampled_probs = np.asarray(sampled, dtype=np.float32)
    return Output(
        np.asarray(tokens, dtype=np.int32),
        logits,
        logprobs,
        np.log(sampled_probs),
        sampled_probs,
    )


def compare(reference, actual, name):
    differences, first_differences = {}, {}
    for field in COMPARED_FIELDS:
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


def real_rows(batch):
    return [row for row in batch["requests"] if not row.get("is_fake")]


def aligned_peer_load(target, peer, expected_ids):
    if peer is None or target["forward_step"] != peer["forward_step"]:
        return False
    common_begin = max(target["forward_start_us"], peer["forward_start_us"])
    common_end = min(target["forward_end_us"], peer["forward_end_us"])
    rows = real_rows(peer)
    return (
        common_begin < common_end
        and len(rows) == len(expected_ids)
        and {row.get("rpc_request_id") for row in rows} == expected_ids
        and all(row.get("q_len", 0) > 0 for row in rows)
    )


def forward_evidence(batch):
    rows = real_rows(batch)
    return {
        "dp_rank": batch["dp_rank"],
        "exec_id": batch["exec_id"],
        "forward_us": [batch["forward_start_us"], batch["forward_end_us"]],
        "real_requests": len(rows),
        "real_tokens": sum(row["q_len"] for row in rows),
        "requests": rows,
    }


def evidence(
    log_dir,
    offsets,
    *,
    neighbors=(),
    target_first=True,
    chunked=False,
    require_dp_rank=False,
    target_rank=0,
    target_rpc_id=None,
    peer_rpc_ids=(),
):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        targets = defaultdict(list)
        forwards = {}
        for batch in read_records(log_dir, offsets):
            dp_rank = batch["dp_rank"] if require_dp_rank else 0
            if peer_rpc_ids and "forward_step" in batch:
                key = (dp_rank, batch["forward_step"])
                assert key not in forwards, (key, batch)
                forwards[key] = batch
            rows = real_rows(batch)
            for i, row in enumerate(rows):
                if row["input_len"] == TARGET_LENGTH and (
                    target_rpc_id is None or row.get("rpc_request_id") == target_rpc_id
                ):
                    targets[(dp_rank, row["request_id"])].append((batch, i, rows))
        matches, candidates = [], {}
        for (dp_rank, request_id), records in targets.items():
            target_prefill = []
            mixed = {"prefill": [], "decode": []}
            aligned = {}
            for batch, i, rows in records:
                row = rows[i]
                phase = row["phase"]
                if phase == "prefill":
                    target_prefill.append((row.get("kv_len", 0), row.get("q_len", 0)))
                wanted_row = 0 if target_first else len(rows) - 1
                lengths = {peer["input_len"] for peer in rows if peer["phase"] == phase}
                if neighbors and i == wanted_row and set(neighbors).issubset(lengths):
                    mixed[phase].append(
                        (batch["exec_id"], i, [peer["input_len"] for peer in rows])
                    )
                    if peer_rpc_ids and "forward_step" in batch:
                        peer = forwards.get((1, batch["forward_step"]))
                        if aligned_peer_load(batch, peer, peer_rpc_ids):
                            aligned.setdefault(phase, (batch, peer))
            candidates[(dp_rank, request_id)] = (target_prefill, mixed, list(aligned))
            alone = all(len(rows) == 1 for _, _, rows in records)
            saw_cut = any(prefix > 0 for prefix, _ in target_prefill)
            if (
                sum(n for _, n in target_prefill) == TARGET_LENGTH
                and saw_cut == chunked
                and (all(mixed.values()) if neighbors else alone)
                and (not peer_rpc_ids or aligned.keys() == mixed.keys())
            ):
                matches.append((dp_rank, request_id, target_prefill, mixed, aligned))
        # The logger can append a previous request's last decode after the
        # offset. A match must carry this request's complete prefill history.
        if matches:
            ((dp_rank, request_id, target_prefill, mixed, aligned),) = matches
            assert dp_rank == target_rank, (dp_rank, target_rank, request_id)
            print(
                f"target DP rank {dp_rank}, recorded request {request_id}", flush=True
            )
            print(
                f"target prefill (prefix, grant) {sorted(target_prefill)}", flush=True
            )
            for name, records in mixed.items():
                if records:
                    print(
                        f"mixed {name} (exec id, target row, prompt lengths) {records}",
                        flush=True,
                    )
            for phase, (batch, peer) in aligned.items():
                print(
                    json.dumps(
                        {
                            "event": "aligned_dp_forward",
                            "phase": phase,
                            "forward_step": batch["forward_step"],
                            "target_request_id": request_id,
                            "target": forward_evidence(batch),
                            "peer": forward_evidence(peer),
                        }
                    ),
                    flush=True,
                )
            return
        time.sleep(0.1)
    raise AssertionError(
        f"missing execution evidence for neighbors {neighbors}, "
        f"target_first={target_first}, chunked={chunked}: "
        f"{candidates=}, {peer_rpc_ids=}"
    )


def mixed_request(
    stub,
    blocker,
    order,
    target,
    steps,
    vocab_size,
    *,
    target_request=None,
    peer_batch=None,
):
    # Queue a rank-1 batch with a longer decode lifetime than the local blocker.
    # The recorder checks that the side load actually survives target admission.
    hold = start_request(stub, blocker, BLOCKER_STEPS)
    next(hold)
    if peer_batch is not None:
        peer_stub, peer_inputs = peer_batch
        peer_call = peer_stub.BatchGenerateCall.future(
            pb.BatchGenerateInputPB(inputs=peer_inputs), timeout=180
        )
    calls = []
    for ids in order:
        if ids is target and target_request is not None:
            calls.append(stub.GenerateStreamCall(target_request, timeout=180))
        else:
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
    if peer_batch is not None:
        replies = peer_call.result().results
        assert len(replies) == len(peer_inputs)
        for reply in replies:
            assert reply.HasField("final_output") and not reply.HasField(
                "error_info"
            ), reply.error_info
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--address", default="127.0.0.1:19481")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--save", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--chunked", action="store_true")
    parser.add_argument(
        "--cross-dp-only",
        action="store_true",
        help="Run solo and the two cross-DP load cases",
    )
    parser.add_argument(
        "--dp-addresses", nargs="+", help="RPC addresses in DP rank order"
    )
    parser.add_argument(
        "--save-cases", type=Path, help="Save raw target RPC arrays for each case"
    )
    smoke = parser.add_mutually_exclusive_group()
    smoke.add_argument("--greedy-smoke", "--off-smoke", action="store_true")
    smoke.add_argument("--greedy-logits", action="store_true")
    args = parser.parse_args()
    if args.cross_dp_only and (not args.dp_addresses or args.chunked or args.reference):
        parser.error(
            "--cross-dp-only requires a DP address list and a fresh budget-0 solo reference"
        )
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    vocab_size = read_vocab_size(args.checkpoint)
    fixed = tokenizer.encode(CONTINUATION, add_special_tokens=False).ids
    target = make_prompt(tokenizer, TARGET_LENGTH, " The ocean is calm.")
    short = make_prompt(tokenizer, SHORT_LENGTH, " Good.")
    long = make_prompt(tokenizer, LONG_LENGTH, " An oak grew near the hill.")
    blocker = make_prompt(tokenizer, 37, " Write a long story.")
    args.log_dir = args.log_dir.resolve()
    with ExitStack() as channels:
        addresses = args.dp_addresses or [args.address]
        stubs = []
        for address in addresses:
            channel = channels.enter_context(grpc.insecure_channel(address))
            grpc.channel_ready_future(channel).result(timeout=600)
            stubs.append(RpcServiceStub(channel))
        stub = stubs[0]
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

        require_dp_rank = bool(args.dp_addresses)
        if args.save_cases:
            args.save_cases.mkdir(parents=True, exist_ok=True)

        def compare_case(reference, actual, name):
            if args.save_cases:
                np.savez(
                    args.save_cases / f"{name.replace(' ', '-')}.npz", **vars(actual)
                )
            compare(reference, actual, name)

        offsets = record_offsets(args.log_dir)
        solo = collect(
            start_request(stub, target, len(fixed), measure=True),
            len(fixed),
            vocab_size,
        )
        evidence(
            args.log_dir, offsets, chunked=args.chunked, require_dp_rank=require_dp_rank
        )
        baseline = solo
        if args.reference:
            with np.load(args.reference) as reference:
                baseline = Output(
                    **{field: reference[field] for field in COMPARED_FIELDS}
                )
            compare_case(baseline, solo, "cross-budget solo")
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
        for order, target_first, neighbors in () if args.cross_dp_only else cases:
            offsets = record_offsets(args.log_dir)
            actual = mixed_request(stub, blocker, order, target, len(fixed), vocab_size)
            evidence(
                args.log_dir,
                offsets,
                neighbors=neighbors,
                target_first=target_first,
                chunked=args.chunked,
                require_dp_rank=require_dp_rank,
            )
            compare_case(
                baseline, actual, f"target {'first' if target_first else 'last'}"
            )

        if require_dp_rank and not args.chunked:
            assert (
                len(stubs) >= 3
            ), "DP rank comparison needs two target ranks and a peer"
            for owner, peer_prompts in ((0, (short,)), (2, (long, short, short))):
                offsets = record_offsets(args.log_dir)
                target_request = make_request(target, len(fixed), measure=True)
                peer_inputs = [
                    make_request(prompt, PEER_STEPS, streaming=False)
                    for prompt in peer_prompts
                ]
                actual = mixed_request(
                    stubs[owner],
                    blocker,
                    (target, short, long),
                    target,
                    len(fixed),
                    vocab_size,
                    target_request=target_request,
                    peer_batch=(stubs[1], peer_inputs),
                )
                evidence(
                    args.log_dir,
                    offsets,
                    neighbors=(SHORT_LENGTH, LONG_LENGTH),
                    require_dp_rank=require_dp_rank,
                    target_rank=owner,
                    target_rpc_id=str(target_request.request_id),
                    peer_rpc_ids={str(request.request_id) for request in peer_inputs},
                )
                compare_case(baseline, actual, f"target DP rank {owner}")


if __name__ == "__main__":
    main()
