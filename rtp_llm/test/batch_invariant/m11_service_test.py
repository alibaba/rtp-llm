"""Compare M11 token IDs, full-vocabulary logits and sampled probabilities on DP8."""

import json
import os
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import grpc
import numpy as np
import requests

from rtp_llm.config.py_config_modules import ServerConfig
from rtp_llm.frontend.tokenizer_factory.tokenizer_factory import TokenizerFactory
from rtp_llm.test.batch_invariant import service_test as rpc
from rtp_llm.test.smoke.qwen4_m11_smoke_test import Qwen4M11SmokeTest
from rtp_llm.test.utils.maga_server_manager import MagaServerManager
from rtp_llm.utils.qwen4_ple import shared_ple


def verify_layout(
    directory, offsets, rank, request_id, grants, row_order, decode_steps, peer_id=None
):
    """Match this RPC to its actual grants and at least one forward in each phase."""
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        records = list(rpc.read_records(directory, offsets))
        peer = {
            batch["forward_step"]: batch
            for batch in records
            if batch.get("dp_rank") == 1
            and any(
                row.get("rpc_request_id") == peer_id for row in rpc.real_rows(batch)
            )
        }
        prefill = []
        phases = set()
        solo = len(row_order) == 1
        target_steps = set()
        decodes = 0
        for batch in records:
            if batch.get("dp_rank") != rank:
                continue
            rows = rpc.real_rows(batch)
            for row in rows:
                if row.get("rpc_request_id") != request_id:
                    continue
                target_steps.add(batch["forward_step"])
                if row["phase"] == "prefill":
                    prefill.append((row["kv_len"], row["q_len"]))
                elif row["phase"] == "decode":
                    decodes += 1
                lengths = [r["input_len"] for r in rows if r["phase"] == row["phase"]]
                if lengths != row_order:
                    continue
                if peer_id is not None:
                    other = peer.get(batch["forward_step"])
                    if not other or (
                        max(batch["forward_start_us"], other["forward_start_us"])
                        >= min(batch["forward_end_us"], other["forward_end_us"])
                    ):
                        continue
                phases.add(row["phase"])

        if (
            sorted(prefill) == grants
            and decodes == decode_steps
            and phases == {"prefill", "decode"}
        ):
            if solo:
                for batch in records:
                    if batch.get("forward_step") in target_steps:
                        ids = [r.get("rpc_request_id") for r in rpc.real_rows(batch)]
                        expected = [request_id] if batch["dp_rank"] == rank else []
                        assert ids == expected, (batch["dp_rank"], ids, expected)
            return
        time.sleep(0.1)
    raise AssertionError(
        f"Missing schedule evidence for RPC {request_id} on rank {rank}, "
        f"observed prefill {sorted(prefill)}, phases {phases}, decode steps {decodes}, "
        f"expected {grants}, {row_order}, {decode_steps}"
    )


def run_cases(server, directory, checkpoint, budget, baseline):
    tokenizer = TokenizerFactory.create(str(checkpoint), str(checkpoint), "qwen4_exp")

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    fixed = encode(rpc.CONTINUATION)
    vocab_size = rpc.read_vocab_size(checkpoint)
    target = rpc.make_prompt(encode, 257, " The ocean is calm.")
    short = rpc.make_prompt(encode, 11, " Good.")
    long = rpc.make_prompt(encode, 321, " An oak grew near the hill.")
    blocker = rpc.make_prompt(encode, 37, " Write a long story.")
    config = ServerConfig()
    config.start_port = int(server.port)
    with ExitStack() as stack:
        stubs = []
        for rank in range(8):
            config.set_local_rank(rank)
            address = f"127.0.0.1:{config.rpc_server_port}"
            channel = stack.enter_context(grpc.insecure_channel(address))
            grpc.channel_ready_future(channel).result(timeout=600)
            stubs.append(rpc.RpcServiceStub(channel))

        def check(rank, prompts, grants, *, busy=False):
            offsets = rpc.record_offsets(directory)
            request = rpc.make_request(target, len(fixed), measure=True)
            peer = rpc.make_request(short, 128) if busy else None
            if len(prompts) == 1:
                call = stubs[rank].GenerateStreamCall(request, timeout=180)
                actual = rpc.collect(call, len(fixed), vocab_size)
            else:
                actual = rpc.run_mixed(
                    stubs[rank],
                    blocker,
                    prompts,
                    target,
                    len(fixed),
                    vocab_size,
                    target_request=request,
                    blocker_steps=2,
                    peer=(stubs[1], peer) if busy else None,
                )
            np.testing.assert_array_equal(actual["tokens"], fixed)
            order = list(map(len, prompts))
            verify_layout(
                directory,
                offsets,
                rank,
                str(request.request_id),
                grants,
                order,
                len(fixed) - 1,
                str(peer.request_id) if busy else None,
            )
            if baseline is not None:
                case = f"budget={budget} rank={rank} order={order} busy={busy}"
                rpc.compare(baseline, actual, case)
            return actual

        grants = [(0, 256), (256, 1)] if budget else [(0, 257)]
        solo = check(0, [target], grants)
        if baseline is None:
            baseline = solo
        if budget:
            check(0, [target, short], grants)
            check(0, [short, target], [(0, 128), (128, 129)])
        else:
            check(7, [target], grants)
            check(0, [target, short, long], grants)
            check(0, [long, short, target], grants)
            check(7, [target, short, long], grants, busy=True)
        return baseline


def main():
    checkpoint = Path(os.environ["QWEN4_M11_CHECKPOINT"]).resolve(strict=True)
    config = json.loads((checkpoint / "config.json").read_text())
    assert config["quantization_config"]["quant_method"] == "mxfp4"
    output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]).resolve()
    with shared_ple(checkpoint) as table_fd:
        run_budgets(checkpoint, config["text_config"], output, table_fd)


def run_budgets(checkpoint, text_config, output, table_fd):
    baseline = None
    for budget in (0, 256):
        run_dir = output / f"budget-{budget}"
        run_dir.mkdir(parents=True, exist_ok=True)
        settings = {
            "RTP_BATCH_INVARIANT": "1",
            "MAGA_SERVER_WORK_DIR": str(run_dir),
            "TEST_UNDECLARED_OUTPUTS_DIR": str(run_dir),
        }
        with patch.dict(os.environ, settings):
            smoke_args = os.environ["SMOKE_ARGS"] + f" --prefill_chunk_size {budget}"
            server = MagaServerManager(
                env_args={"FT_SERVER_TEST": "1"},
                role_name="m11_batch_invariant",
                smoke_args_str=smoke_args,
            )
            try:
                if not server.start_server(
                    str(checkpoint), "qwen4_exp", timeout=1800, pass_fds=(table_fd,)
                ):
                    raise RuntimeError(
                        f"M11 server failed with prefill budget {budget}"
                    )
                baseline = run_cases(
                    server, run_dir / "logs", checkpoint, budget, baseline
                )
                if budget == 0:
                    np.savez(run_dir / "solo.npz", **baseline)
                    with requests.Session() as session:
                        session.trust_env = False
                        smoke = Qwen4M11SmokeTest()
                        smoke.session = session
                        smoke.server_url = f"http://127.0.0.1:{server.port}/"
                        smoke.text_config = text_config
                        smoke.mtp_enabled = False
                        smoke.test_chinese_answer()
                        smoke.test_qsa_beyond_indexer_budget()
            finally:
                if not server.stop_server():
                    raise RuntimeError(
                        f"M11 workers still active after budget {budget}"
                    )
        manifests = [
            json.loads(path.read_text())
            for path in run_dir.rglob("batch_record/owner-*/manifest.json")
        ]
        assert {m["dp_rank"] for m in manifests} == set(range(8)), manifests
        for manifest in manifests:
            assert manifest["engine"]["record_decode_sample_rate"] == 1, manifest
            assert manifest["dropped"] == 0 and manifest["errors"] == 0, manifest


if __name__ == "__main__":
    main()
