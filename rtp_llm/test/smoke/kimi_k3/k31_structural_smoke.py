#!/usr/bin/env python3
"""Run explicit-token synthetic K3.1 PD requests through the existing group RPC.

This checks completion and serving metadata. It does not compare language,
logits or model accuracy; cache/owner/collective events require a separate audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from .text_smoke import prefill_cache_tiers
else:
    from text_smoke import prefill_cache_tiers


@dataclass(frozen=True)
class Case:
    name: str
    length: int
    seed: int
    owner: int = 0
    reuse: bool = False
    minimum_reuse: int = 0
    force_accept: bool = False
    explicit_tokens: tuple[int, ...] | None = None


def tokens(case):
    # Prefixes are stable across lengths, and ordinary IDs avoid control tokens.
    if case.explicit_tokens is not None:
        if len(case.explicit_tokens) != case.length or any(
            type(token) is not int or token < 0 for token in case.explicit_tokens
        ):
            raise ValueError("explicit token fixture must match the declared length")
        return list(case.explicit_tokens)
    return [100 + ((i * 48271 + case.seed * 7919) % 32000)
            for i in range(case.length)]


def check_draft_execution(rounds, *, target_only, force_accept=False):
    """Keep the original MTP gate; target-only is an explicit separate mode."""
    if target_only:
        if force_accept or rounds != 0:
            raise RuntimeError("target-only request cannot force or execute draft")
    elif rounds < 1:
        raise RuntimeError("no MTP draft execution")


def run_group(args, cases, output):
    import grpc
    import torch
    from google.protobuf.json_format import MessageToDict
    from rtp_llm.config.generate_config import GenerateConfig, RoleAddr, ThinkingMode
    from rtp_llm.cpp.model_rpc.model_rpc_client import trans_input
    from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
        CancelRequestPB, EnqueueGroupRequestPB, FetchRequestPB,
    )
    from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub
    from rtp_llm.ops import RoleType
    from rtp_llm.utils.base_model_datatypes import GenerateInput, RequestInfo
    from rtp_llm.utils.grpc_util import trans_tensor

    # PD handoffs arrive independently. Keep the concurrency group alive long
    # enough for all owner streams to meet in one real Decode batch. The event
    # auditor still requires that batch; duration alone is not evidence.
    target_only = bool(getattr(args, 'target_only', False))
    diagnostic_tensors = bool(getattr(args, 'diagnostic_tensors', False))
    if not cases or (target_only and any(case.force_accept for case in cases)):
        raise ValueError("empty group or forced-accept target-only fixture")
    output_tokens = (getattr(args, 'group_output_tokens', None)
                     or (args.batch_output_tokens
                         if all(case.name.startswith('batch32-') for case in cases)
                         else args.output_tokens))
    group_id = time.time_ns() % (2**62)
    group = EnqueueGroupRequestPB(batch_id=group_id, dp_rank=0,
                                  fetch_attach_timeout_ms=args.timeout * 1000)
    inputs = {}
    for offset, case in enumerate(cases):
        owner_port = args.decode_port + case.owner * args.decode_tp * 9
        route = dict(ip=args.decode_ip, http_port=owner_port,
                     grpc_port=owner_port + 1)
        token_ids = tokens(case)
        request_id = group_id + offset
        config = GenerateConfig(
            max_new_tokens=output_tokens, min_new_tokens=output_tokens,
            ignore_eos=True, do_sample=False, top_k=1, top_p=1.0,
            timeout_ms=args.timeout * 1000, random_seed=3101, aux_info=True,
            thinking_mode=ThinkingMode.DISABLED, max_thinking_tokens=0,
            reuse_cache=case.reuse, force_sp_accept=case.force_accept,
            return_logits=diagnostic_tensors, return_hidden_states=diagnostic_tensors,
            role_addrs=[RoleAddr(role=RoleType.DECODE, **route)],
        )
        item = GenerateInput(
            request_id=request_id, token_ids=torch.tensor(token_ids, dtype=torch.int32),
            mm_inputs=[], generate_config=config,
            request_info=RequestInfo(request_id=case.name),
        )
        group.requests.add().input.CopyFrom(trans_input(item))
        inputs[request_id] = (case, token_ids, route)
        (output / f"{case.name}.request.json").write_text(json.dumps(dict(
            case=case.__dict__, request_id=request_id,
            requested_route=route, input_tokens=token_ids,
            requested_output_tokens=output_tokens,
            accuracy_scope='synthetic_structure_only',
        ), indent=2) + '\n')

    with grpc.insecure_channel(args.prefill_rpc) as channel:
        stub = RpcServiceStub(channel)
        admitted = set()
        started = time.monotonic()
        start_time_ns = time.time_ns()
        try:
            ack = stub.EnqueueGroup(group, timeout=args.timeout)
            admitted = {item.request_id for item in ack.successes}
            if admitted != set(inputs) or ack.errors:
                raise RuntimeError(f"Group admission mismatch: {ack}")

            def fetch(request_id):
                ids, aux, finished = [], None, False
                tensor_frames = []
                for frame in stub.FetchResponse(
                    FetchRequestPB(request_id=request_id), timeout=args.timeout
                ):
                    flat = frame.flatten_output
                    new_ids = (trans_tensor(flat.output_ids).reshape(-1).tolist()
                               if flat.HasField("output_ids") else [])
                    if diagnostic_tensors:
                        tensors = {'output_offset': len(ids), 'new_token_ids': new_ids,
                                   'output_end': len(ids) + len(new_ids)}
                        for field in ('logits', 'hidden_states'):
                            if flat.HasField(field):
                                tensors[field] = trans_tensor(getattr(flat, field)).clone()
                        tensor_frames.append(tensors)
                    ids.extend(new_ids)
                    if flat.aux_info:
                        aux = flat.aux_info[0]
                    finished = bool(flat.finished and flat.finished[0])
                case, token_ids, route = inputs[request_id]
                if diagnostic_tensors:
                    if not any('logits' in row and 'hidden_states' in row for row in tensor_frames):
                        raise RuntimeError(f'{case.name}: requested diagnostic tensors missing')
                    torch.save(tensor_frames, output / f'{case.name}.tensors.pt')
                if not finished or aux is None or len(ids) != output_tokens:
                    raise RuntimeError(f"{case.name}: incomplete or wrong output length: "
                                       f"finished={finished}, tokens={len(ids)}")
                if not aux.pd_sep or aux.input_len != case.length:
                    raise RuntimeError(f"{case.name}: missing PD or input length mismatch")
                check_draft_execution(
                    aux.speculative_draft_rounds,
                    target_only=target_only,
                    force_accept=case.force_accept,
                )
                aux_dict = MessageToDict(aux, preserving_proto_field_name=True)
                aux_dict["speculative_metrics"] = dict(aux.speculative_metrics)
                for field in ('prefill_total_reuse_len', 'prefill_local_reuse_len',
                              'prefill_memory_reuse_len', 'prefill_disk_reuse_len'):
                    aux_dict[field] = int(getattr(aux, field))
                tiers = prefill_cache_tiers(aux_dict)
                if tiers['total'] < case.minimum_reuse:
                    raise RuntimeError(f"{case.name}: insufficient reuse: {tiers}")
                record = dict(
                    case=case.__dict__, request_id=request_id, requested_owner=case.owner,
                    requested_route=route, input_tokens=token_ids, output_tokens=ids,
                    requested_output_tokens=output_tokens,
                    input_sha256=hashlib.sha256(
                        torch.tensor(token_ids, dtype=torch.int32).numpy().tobytes()
                    ).hexdigest(), aux_info=aux_dict, cache_tiers=tiers,
                    rpc_finished=True, structural_passed=True,
                    elapsed_group_seconds=time.monotonic() - started,
                    start_time_ns=start_time_ns, end_time_ns=time.time_ns(),
                    accuracy_scope='synthetic_structure_only',
                    runtime_event_audit_required=True,
                    model_mode='target_only' if target_only else 'synthetic_dsa_mtp',
                    mtp_acceptance='N/A' if target_only else 'separate actual-candidate audit required',
                    diagnostic_tensors=diagnostic_tensors,
                    performance_measurement=False,
                )
                return record

            with ThreadPoolExecutor(max_workers=len(cases)) as pool:
                results = list(pool.map(fetch, inputs))
        except Exception:
            for request_id in admitted:
                try:
                    stub.Cancel(CancelRequestPB(request_id=request_id), timeout=5)
                except grpc.RpcError:
                    pass
            raise
    for result in results:
        (output / f"{result['case']['name']}.json").write_text(
            json.dumps(result, indent=2) + '\n'
        )
        compact = {key: result[key] for key in
                   ('request_id', 'cache_tiers', 'structural_passed')}
        compact['case'] = {key: value for key, value in result['case'].items()
                           if key != 'explicit_tokens'}
        print(json.dumps(compact), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefill-rpc', required=True)
    parser.add_argument('--decode-ip', required=True)
    parser.add_argument('--decode-port', type=int, required=True)
    parser.add_argument('--decode-tp', type=int, choices=(4, 8), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--output-tokens', type=int, default=8)
    parser.add_argument('--batch-output-tokens', type=int, default=128)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--first-request-only', action='store_true')
    parser.add_argument('--target-only', action='store_true',
                        help='Validate ordinary Decode with no speculative draft rounds')
    parser.add_argument('--history-tokens', type=int, default=65536,
                        help='Long-history fixture extent; lower values are explicit reduced-context validation')
    parser.add_argument('--case-group', choices=('all', 'batch32', 'mtp', 'mtp-fixture', 'owners'), default='all')
    args = parser.parse_args()
    if not 8192 <= args.history_tokens <= 65536 or args.history_tokens % 8192:
        parser.error("Long-history extent must be a TP8 PageRR stripe multiple between8192and65536")
    args.output.mkdir(parents=True, exist_ok=False)
    groups = [[Case('first-17', 17, 1)]]
    if not args.first_request_only:
        groups += [
            [Case('kv-2048', 2048, 2), Case('kv-2049', 2049, 3),
             Case('page-padding-1025', 1025, 4)],
            # PageRR prefix lookup uses one complete TP8 stripe: 8 * 1024.
            [Case('prefix-seed-8193', 8193, 5, reuse=True)],
            [Case('prefix-hit-8193', 8193, 5, reuse=True, minimum_reuse=8192)],
            [Case('prefix-partial-8197', 8197, 5, reuse=True, minimum_reuse=8192)],
            [Case('short-q-long-history-seed', args.history_tokens + 1, 6, reuse=True)],
            [Case('short-q-long-history', args.history_tokens + 17, 6, reuse=True, minimum_reuse=args.history_tokens)],
            [Case(f'batch32-{i}', 17 + i % 7, 100 + i,
                  owner=i % (8 // args.decode_tp)) for i in range(32)],
            [Case('mtp-normal-17', 17, 500)],
        ]
        if args.decode_tp == 4:
            groups += [[Case('owner0-concurrent', 2051, 600),
                        Case('owner1-concurrent', 2055, 601, owner=1)],
                       [Case('owner1-directed', 17, 602, owner=1)]]
    if args.case_group == 'mtp-fixture':
        groups = [[Case('constructed-mtp-force-accept',17,500,force_accept=True)]]
    elif args.case_group != 'all':
        prefixes = {'batch32': 'batch32-', 'mtp': 'mtp-', 'owners': 'owner'}
        groups = [group for group in groups if group[0].name.startswith(prefixes[args.case_group])]
        if not groups:
            parser.error('Selected case group is unavailable for this topology')
    results = []
    for group in groups:
        results.extend(run_group(args, group, args.output))
    (args.output / 'structural-summary.json').write_text(json.dumps(dict(
        requests=len(results), decode_tp=args.decode_tp, decode_dp=8 // args.decode_tp,
        case_group=args.case_group, constructed_acceptance_fixture=args.case_group=='mtp-fixture',
        long_history_tokens=args.history_tokens,
        reduced_context_baseline=args.history_tokens < 65536,
        accuracy_scope='synthetic_structure_only', structural_passed=True,
        runtime_event_audit_required=True, performance_validated=False,
    ), indent=2) + '\n')


if __name__ == '__main__':
    main()
