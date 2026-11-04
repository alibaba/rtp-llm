"""Fixed-continuation native PPU capture against an existing lossless fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

TASK_PREFIX = "__rtp_v41_acceptance__/"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as reader:
        for chunk in iter(lambda: reader.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def save_json(path, value):
    with Path(path).open("x") as writer:
        json.dump(value, writer, ensure_ascii=False, indent=2)
        writer.write("\n")


def load_fixture(path):
    fixture = json.loads(Path(path).read_text())
    if fixture["schema_version"] != 1 or fixture["vocab_size"] <= 0:
        raise ValueError("invalid fixed continuation fixture")
    for case, entry in fixture["cases"].items():
        if not case or any(
            not c.isascii() or not (c.isalnum() or c in "_-") for c in case
        ):
            raise ValueError("invalid fixture case id")
        for field in ("input_ids", "continuation_tokens"):
            ids = entry[field]
            if not ids or any(
                type(i) is not int or not 0 <= i < fixture["vocab_size"] for i in ids
            ):
                raise ValueError("invalid fixture token IDs")
    return fixture


def prepare(args):
    manifest_path = args.reference / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    quickset = args.reference / manifest["quickset_file"]
    if (
        not manifest["capture_selfcheck_pass"]
        or not manifest["storage_lossless"]
        or digest(quickset) != manifest["quickset_sha256"]
    ):
        raise ValueError("reference manifest or lossless package checksum failed")
    cases = {}
    vocab = None
    with np.load(quickset, allow_pickle=False) as arrays:
        for case in manifest["inputs"]["cases"]:
            logits = arrays[f"{case}__logits_fp32"]
            ids = arrays[f"{case}__input_ids"]
            continuation = arrays[f"{case}__continuation_tokens"]
            contexts = arrays[f"{case}__context_lengths"]
            if (
                logits.dtype != np.float32
                or logits.ndim != 2
                or logits.shape[0] != len(continuation)
                or not np.isfinite(logits).all()
            ):
                raise ValueError(f"invalid reference logits for {case}")
            if vocab is not None and vocab != logits.shape[1]:
                raise ValueError("reference vocabulary differs across cases")
            vocab = logits.shape[1]
            if not np.array_equal(
                contexts, np.arange(len(ids), len(ids) + len(continuation))
            ):
                raise ValueError("reference context history mismatch")
            cases[case] = {
                "input_ids": ids.tolist(),
                "continuation_tokens": continuation.tolist(),
            }
    fixture = {
        "schema_version": 1,
        "vocab_size": vocab,
        "cases": cases,
        "reference_manifest_sha256": digest(manifest_path),
        "reference_quickset_sha256": digest(quickset),
        "reference_inputs_sha256": manifest["inputs"]["sha256"],
        "model_files": manifest["model_files"],
    }
    save_json(args.output, fixture)
    print(
        json.dumps(
            {
                "fixture": str(args.output),
                "sha256": digest(args.output),
                "cases": list(cases),
                "vocab_size": vocab,
            }
        )
    )


def tensor_ids(tensor):
    if not tensor.shape:
        return []
    if tensor.data_type != 1:
        raise ValueError("native output token IDs are not int32")
    values = np.frombuffer(tensor.int32_data, dtype="<i4")
    if values.size != int(np.prod(tensor.shape)):
        raise ValueError("native output ID shape mismatch")
    return values.tolist()


def request(args):
    import grpc

    from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
        GenerateInputPB,
        RoleAddrPB,
    )
    from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub

    if sys.byteorder != "little":
        raise ValueError("native capture requires a little-endian runtime")
    fixture = load_fixture(args.fixture)
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = json.loads((args.runtime / "native-runtime.json").read_text())
    snapshot = json.loads((args.runtime / ".v41_snapshot.json").read_text())
    if any(
        runtime[key] != snapshot[key]
        for key in ("commit", "source_sha256", "snapshot_id")
    ):
        raise ValueError("native runtime and source identities differ")
    chosen = args.case or list(fixture["cases"])
    request_base = time.time_ns() % (1 << 60)
    with grpc.insecure_channel(
        args.prefill, options=[("grpc.max_receive_message_length", 64 << 20)]
    ) as channel:
        stub = RpcServiceStub(channel)
        for index, case in enumerate(chosen):
            entry = fixture["cases"][case]
            request_id = request_base + index
            query = GenerateInputPB(
                request_id=request_id,
                client_id="v41-fixed-continuation",
                group_size=1,
                start_time=int(time.time() * 1e6),
            )
            query.token_ids.extend(entry["input_ids"])
            config = query.generate_config
            config.max_new_tokens = config.min_new_tokens = len(
                entry["continuation_tokens"]
            )
            config.num_beams = config.num_return_sequences = config.top_k = 1
            config.top_p = config.temperature = config.repetition_penalty = 1.0
            config.return_logits = config.is_streaming = config.ignore_eos = (
                config.force_disable_sp_run
            ) = True
            config.return_incremental = True
            config.reuse_cache = False
            config.can_use_pd_separation = not args.single_engine
            config.timeout_ms = int(args.timeout * 1000)
            config.task_id.value = TASK_PREFIX + case
            config.aux_info.value = True
            p_host, p_port = args.prefill.rsplit(":", 1)
            config.role_addrs.append(
                RoleAddrPB(
                    role=RoleAddrPB.PREFILL,
                    ip=p_host,
                    grpc_port=int(p_port),
                    http_port=int(p_port) - 1,
                )
            )
            if not args.single_engine:
                d_host, d_port = args.decode.rsplit(":", 1)
                port = int(d_port) + (index % 8) * 9
                config.role_addrs.append(
                    RoleAddrPB(
                        role=RoleAddrPB.DECODE,
                        ip=d_host,
                        grpc_port=port,
                        http_port=port - 1,
                    )
                )
            out = args.output / case
            out.mkdir()
            (out / "request.pb").write_bytes(query.SerializeToString())
            observed, frames, finished = [], [], False
            started = time.monotonic()
            call = stub.GenerateStreamCall(query, timeout=args.timeout)
            try:
                for response in call:
                    if response.error_info.error_code:
                        raise RuntimeError(str(response.error_info))
                    if response.request_id != request_id:
                        raise ValueError("native response request ID mismatch")
                    flat = response.flatten_output
                    if not flat.aux_info:
                        continue
                    if len(flat.aux_info) != 1 or len(flat.finished) != 1:
                        raise ValueError("capture received multiple sequences")
                    aux = flat.aux_info[0]
                    tokens = tensor_ids(flat.output_ids)
                    total = aux.output_len
                    start = total - len(tokens)
                    if (
                        aux.input_len != len(entry["input_ids"])
                        or start < 0
                        or start > len(observed)
                    ):
                        raise ValueError("native response length/history mismatch")
                    overlap = len(observed) - start
                    if tokens[:overlap] != observed[start:]:
                        raise ValueError("native response token overlap mismatch")
                    observed.extend(tokens[overlap:])
                    if (
                        observed != entry["continuation_tokens"][: len(observed)]
                        or len(observed) != total
                    ):
                        raise ValueError(
                            "native output differs from frozen continuation"
                        )
                    finished = flat.finished[0]
                    if bool(aux.pd_sep) != (not args.single_engine):
                        raise ValueError(
                            "native request used the wrong execution topology"
                        )
                    frames.append(
                        {
                            "output_len": total,
                            "tokens": tokens,
                            "finished": finished,
                            "pd_sep": aux.pd_sep,
                            "logits_shape": list(flat.logits.shape),
                            "logits_dtype": flat.logits.data_type,
                            "raw_logits_pb_sha256": hashlib.sha256(
                                flat.logits.SerializeToString()
                            ).hexdigest(),
                        }
                    )
                    (out / "frames.json").write_text(
                        json.dumps(frames, indent=2) + "\n"
                    )
                if not finished or observed != entry["continuation_tokens"]:
                    raise ValueError("incomplete native fixed continuation")
            finally:
                call.cancel()
            report = {
                "case_id": case,
                "request_id": request_id,
                "finished": finished,
                "output_ids": observed,
                "input_length": len(entry["input_ids"]),
                "fixture_sha256": digest(args.fixture),
                "single_engine": args.single_engine,
                "snapshot_id": snapshot["snapshot_id"],
                "public_commit": snapshot["public_commit"],
                "internal_commit": snapshot["internal_commit"],
                "source_sha256": snapshot["source_sha256"],
                "native_libraries": runtime["libraries"],
                "seconds": time.monotonic() - started,
            }
            save_json(out / "summary.json", report)
            print(
                json.dumps(
                    {
                        key: report[key]
                        for key in (
                            "case_id",
                            "request_id",
                            "finished",
                            "seconds",
                            "snapshot_id",
                        )
                    }
                ),
                flush=True,
            )


def assemble(args):
    fixture = load_fixture(args.fixture)
    arrays, provenance = {}, {}
    identity_keys = ("snapshot_id", "public_commit", "internal_commit", "source_sha256")
    identity = None
    owners = {}
    for directory in args.capture:
        owners[directory] = (
            json.loads((directory.parent / "owner.json").read_text()),
            json.loads((directory.parent / "processes.json").read_text()),
        )
    for summary in sorted(args.requests.glob("*/summary.json")):
        report = json.loads(summary.read_text())
        current_identity = tuple(report[key] for key in identity_keys)
        if (
            not all(current_identity)
            or identity is not None
            and identity != current_identity
        ):
            raise ValueError("mixed or missing committed runtime identities")
        identity = current_identity
        case = report["case_id"]
        entry = fixture["cases"][case]
        if (
            not report["finished"]
            or report["output_ids"] != entry["continuation_tokens"]
            or report["fixture_sha256"] != digest(args.fixture)
        ):
            raise ValueError("request did not verify the complete frozen continuation")
        if case in provenance:
            raise ValueError("duplicate request case")
        rows, records = [], []
        for row, target in enumerate(entry["continuation_tokens"]):
            name = f"{case}-request-{report['request_id']}/row-{row}"
            found = [
                directory / (name + ".json")
                for directory in args.capture
                if (directory / (name + ".json")).exists()
            ]
            if len(found) != 1:
                raise ValueError(
                    f"expected exactly one native raw row: {case}/{row}, got {len(found)}"
                )
            meta_path = found[0]
            metadata = json.loads(meta_path.read_text())
            expected = {
                "schema_version": 1,
                "dtype": "float32",
                "vocab_size": fixture["vocab_size"],
                "row": row,
                "context_length": len(entry["input_ids"]) + row,
                "target_token": target,
                "request_id": report["request_id"],
                "case_id": case,
                "role": "PREFILL" if row == 0 or report["single_engine"] else "DECODE",
                "source": "native_lm_head_before_sampling",
            }
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise ValueError(f"native raw row metadata mismatch: {case}/{row}")
            directory = next(
                directory
                for directory in args.capture
                if meta_path.is_relative_to(directory)
            )
            owner, processes = owners[directory]
            if (
                tuple(owner[key] for key in identity_keys) != identity
                or owner["role"].upper() != expected["role"]
            ):
                raise ValueError("capture owner identity or role differs from request")
            if str(metadata["pid"]) not in processes:
                raise ValueError("native logits came from an unowned process")
            data = meta_path.with_suffix(".f32")
            values = np.fromfile(data, dtype="<f4")
            if (
                values.shape != (fixture["vocab_size"],)
                or not np.isfinite(values).all()
            ):
                raise ValueError(
                    f"native raw row shape or finiteness failed: {case}/{row}"
                )
            rows.append(values)
            records.append(
                {"path": str(data), "sha256": digest(data), "metadata": metadata}
            )
        arrays[f"{case}__logits_fp32"] = np.stack(rows)
        arrays[f"{case}__input_ids"] = np.asarray(entry["input_ids"], dtype=np.int32)
        arrays[f"{case}__continuation_tokens"] = np.asarray(
            entry["continuation_tokens"], dtype=np.int64
        )
        arrays[f"{case}__context_lengths"] = np.arange(
            len(entry["input_ids"]), len(entry["input_ids"]) + len(rows), dtype=np.int32
        )
        provenance[case] = {"request": report, "rows": records}
    if not provenance:
        raise ValueError("no complete native requests to assemble")
    with args.output.open("xb") as writer:
        np.savez_compressed(writer, **arrays)
    save_json(
        args.output.with_suffix(".provenance.json"),
        {
            "schema_version": 1,
            "fixture_sha256": digest(args.fixture),
            "candidate_sha256": digest(args.output),
            "cases": provenance,
            "numerical_verdict": "REQUIRES_COMPARISON",
        },
    )
    print(
        json.dumps(
            {
                "candidate": str(args.output),
                "sha256": digest(args.output),
                "cases": list(provenance),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--reference", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.set_defaults(run=prepare)
    req = commands.add_parser("request")
    req.add_argument("--fixture", type=Path, required=True)
    req.add_argument(
        "--runtime", type=Path, default=Path(__file__).resolve().parents[3]
    )
    req.add_argument("--prefill", required=True, help="native P gRPC endpoint")
    req.add_argument(
        "--decode",
        required=True,
        help="D rank-0 gRPC endpoint; cases rotate across owners",
    )
    req.add_argument("--output", type=Path, required=True)
    req.add_argument("--case", action="append")
    req.add_argument("--timeout", type=float, default=1800)
    req.add_argument("--single-engine", action="store_true")
    req.set_defaults(run=request)
    pack = commands.add_parser("assemble")
    pack.add_argument("--fixture", type=Path, required=True)
    pack.add_argument("--requests", type=Path, required=True)
    pack.add_argument("--capture", type=Path, action="append", required=True)
    pack.add_argument("--output", type=Path, required=True)
    pack.set_defaults(run=assemble)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
