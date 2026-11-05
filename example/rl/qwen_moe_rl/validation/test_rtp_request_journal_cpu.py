#!/usr/bin/env python3
"""Actual TokenRpcClient/protobuf/strict MRO through a localhost CPU fake RPC.

No Torch/native model import or GPU request is needed. The only server created
is this test's grpc.aio fake service on an ephemeral 127.0.0.1 port.
"""
import argparse
import ast
import asyncio
import base64
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import time


async def run(args):
    sys.path.insert(0, str(args.rtp_source.resolve()))
    sys.path.insert(0, str((args.project_root / "adapter").resolve()))
    import grpc
    from rtp_llm.RL import token_rpc_client as actual
    from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb
    from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2_grpc as rpc
    from rtp_request_journal import NativeRpcJournal, JournaledTokenRpcMixin

    adapter_path = args.project_root / "adapter/rtp_multinode_server.py"
    parsed = ast.parse(adapter_path.read_text())
    strict = next(node for node in ast.walk(parsed) if isinstance(node, ast.ClassDef)
                  and node.name == "StrictTokenRpcClient")
    namespace = dict(TokenRpcClient=actual.TokenRpcClient, TokenProtocolError=actual.TokenProtocolError)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[strict], type_ignores=[])),
                 str(adapter_path) + ":actual-strict-client", "exec"), namespace)

    class Observed(JournaledTokenRpcMixin, namespace["StrictTokenRpcClient"]):
        pass

    class Service(rpc.RpcServiceServicer):
        def __init__(self):
            self.active = self.peak = 0
            self.requests = {}
            self.entered = {}
            self.exited = {}
            self.release = asyncio.Event()

        async def GenerateStreamCall(self, request, context):
            request_id = request.request_id
            self.requests[request_id] = request.SerializeToString(deterministic=True)
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.entered.setdefault(request_id, asyncio.Event()).set()
            try:
                if request_id == 301:
                    await self.release.wait()
                else:
                    await asyncio.sleep(.003)
                if request_id == 201:
                    await context.abort(grpc.StatusCode.UNAVAILABLE, "deliberate CPU test error")
                response = pb.GenerateOutputsPB(request_id=request_id,
                    model_version="cpu-policy:1", global_step=7)
                output = response.flatten_output
                output.finished.append(True)
                output.output_ids.CopyFrom(pb.TensorPB(data_type=pb.TensorPB.INT32,
                    shape=[1], int32_data=struct.pack("<i", 5)))
                aux = output.aux_info.add(input_len=len(request.token_ids), output_len=1)
                if request_id != 202:
                    aux.raw_logprobs.CopyFrom(pb.TensorPB(data_type=pb.TensorPB.FP32,
                        shape=[1], fp32_data=struct.pack("<f", -.25)))
                config = request.generate_config
                if config.return_target_logprob:
                    count = config.prompt_logits_end - config.prompt_logits_start
                    logits = output.prompt_logits
                    logits.start_pos = config.prompt_logits_start
                    logits.end_pos = config.prompt_logits_end
                    logits.target_logprobs.CopyFrom(pb.TensorPB(data_type=pb.TensorPB.FP32,
                        shape=[count], fp32_data=struct.pack(f"<{count}f", *([-.25] * count))))
                yield response
            finally:
                self.active -= 1
                self.exited.setdefault(request_id, asyncio.Event()).set()

    async def wait_event(mapping, request_id):
        async def wait():
            while request_id not in mapping:
                await asyncio.sleep(0)
            await mapping[request_id].wait()
        await asyncio.wait_for(wait(), 5)

    service = Service()
    server = grpc.aio.server()
    rpc.add_RpcServiceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    if not port:
        raise RuntimeError("CPU fake service could not bind an ephemeral localhost port")
    await server.start()
    cases = []
    with tempfile.TemporaryDirectory(prefix="rtp-request-journal-cpu-") as tmp:
        journal = NativeRpcJournal(tmp, 1, lambda: dict(native_boot=0, native_incarnation="cpu-test"))
        client = Observed(f"127.0.0.1:{port}", vocab_size=248320, eos_token_ids=[2],
                          timeout=1500, request_journal=journal, max_active_requests=1)
        params = actual.TokenSamplingParams(max_new_tokens=1, temperature=1, top_p=1,
            top_k=0, seed=123, stop_token_ids=(2,), logprobs=True)
        try:
            normal = [client.generate(i, [41, i], params, expected_version="cpu-policy:1")
                      for i in range(101, 107)]
            score = client.score_tokens(401, [31, 32], [5, 6], expected_version="cpu-policy:1")
            results = await asyncio.wait_for(asyncio.gather(*normal, score), 10)
            assert all(result.log_probs == [-.25] for result in results[:-1])
            assert results[-1] == [-.25, -.25]
            assert service.peak == 1
            cases.append("actual_generate_and_score_share_gate1_no_double_acquire")

            request = pb.GenerateInputPB.FromString(service.requests[101])
            cfg = request.generate_config
            assert (cfg.max_new_tokens, cfg.temperature, cfg.top_p, cfg.top_k, cfg.do_sample) == (1, 1, 1, 0, True)
            assert cfg.random_seed.value == 123 and cfg.timeout_ms == 1500000
            assert list(cfg.stop_words_list.rows[0].values) == [2]
            cases.append("actual_protobuf_sampling_seed_timeout_serialization")

            await client.generate(501, [11], params, expected_version=None)
            cases.append("cold_null_expected_version")

            try:
                await client.generate(203, [11], params, expected_version="stale-policy:0")
            except actual.TokenProtocolError:
                pass
            else:
                raise AssertionError("actual TokenRpcClient did not reject stale expected_version")
            cases.append("post_rpc_actual_version_validation_failure")

            try:
                await client.generate(202, [11], params, expected_version="cpu-policy:1")
            except actual.TokenProtocolError as error:
                assert "raw logprobs" in str(error)
            else:
                raise AssertionError("actual nested strict MRO accepted missing raw FP32 logprobs")
            cases.append("actual_nested_strict_raw_lp_failure")

            try:
                await client.generate(201, [11], params, expected_version="cpu-policy:1")
            except grpc.aio.AioRpcError as error:
                assert error.code() == grpc.StatusCode.UNAVAILABLE
            else:
                raise AssertionError("actual grpc transport failure was swallowed")
            await client.generate(204, [12], params, expected_version="cpu-policy:1")
            cases.append("actual_grpc_error_releases_gate")

            active = asyncio.create_task(client.generate(301, [17], params, expected_version="cpu-policy:1"))
            await wait_event(service.entered, 301)
            queued = asyncio.create_task(client.generate(302, [18], params, expected_version="queued-policy"))
            async def queued_is_durable():
                while not any(json.loads(line).get("native_request_id") == "302"
                              for line in journal.path.read_text().splitlines()):
                    await asyncio.sleep(0)
            await asyncio.wait_for(queued_is_durable(), 5)
            queued.cancel()
            try:
                await queued
            except asyncio.CancelledError:
                pass
            active.cancel()
            try:
                await active
            except asyncio.CancelledError:
                pass
            await wait_event(service.exited, 301)
            assert 302 not in service.requests
            await client.generate(303, [19], params, expected_version="cpu-policy:1")
            cases.append("actual_active_and_queued_cancellation_release_gate")

            rows = [json.loads(line) for line in journal.path.read_text().splitlines()]
            by_id = {}
            for row in rows:
                by_id.setdefault(int(row["native_request_id"]), []).append(row)
            for request_id, group in by_id.items():
                assert group[0]["event"] == "queued" and group[-1]["event"] == "finish"
                expected = "cancelled" if request_id in (301, 302) else "error" if request_id in (201, 202, 203) else "success"
                assert group[-1]["outcome"] == expected, (request_id, group[-1])
                assert len({row["journal_id"] for row in group}) == 1
            assert [row["event"] for row in by_id[302]] == ["queued", "finish"]
            assert by_id[302][-1]["started_unix"] is None
            assert by_id[501][0]["expected_model_version"] is None
            assert by_id[401][0]["kind"] == "score_tokens"
            assert by_id[401][0]["prompt_token_ids"] == [31, 32, 5, 6]
            cfg_record = pb.GenerateConfigPB.FromString(base64.b64decode(by_id[101][0]["config_at_request_entry_pb_b64"]))
            assert cfg_record.random_seed.value == 123 and cfg_record.do_sample
            assert by_id[101][1]["sampling"]["seed"] == 123
            assert by_id[101][-1]["response"]["model_version"] == "cpu-policy:1"
            cases.append("actual_protobuf_journal_rows_and_terminal_outcomes")
            receipt = dict(schema="rtp-request-journal-actual-cpu-v1", passed=True,
                timestamp=time.time(), python=sys.version, cases=cases, count=len(cases),
                fake_rpc_received=len(service.requests), fake_rpc_peak_active=service.peak,
                journal_rows=len(rows), used_gpu=False, external_model_requests=False,
                actual_token_client_path=actual.__file__,
                actual_token_client_sha256=hashlib.sha256(Path(actual.__file__).read_bytes()).hexdigest(),
                actual_proto_path=pb.__file__, actual_proto_sha256=hashlib.sha256(Path(pb.__file__).read_bytes()).hexdigest(),
                strict_adapter_sha256=hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
                project_root=str(args.project_root.resolve()), rtp_source=str(args.rtp_source.resolve()))
            print(json.dumps(receipt, indent=2))
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(receipt, indent=2) + "\n")
        finally:
            await client.close()
            await server.stop(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtp-source", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
