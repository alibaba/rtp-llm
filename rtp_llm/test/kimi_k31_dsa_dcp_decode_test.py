"""K3.1 DSA decode across a DCP group: Indexer score -> top-k -> sparse MLA -> V-up -> combine.

Deployment geometry (KV pages of 1024 tokens round-robin over the TP/DCP group, kernel pages of
128, top-k 2048, 32x128 BF16 Indexer, 96 MLA heads of 576/512, V head 128, MTP verify with Q query
tokens per request), requests of mixed context lengths, and every request's owner pages scattered
over a serving-sized page pool. History is procedural (kimi_k31_dsa_fixture), so every rank
computes an fp32 single-GPU reference of its own heads over the global history on demand.

One process runs the whole schedule like a decode server: the workspace owner and the Indexer op
persist, every step builds its Page-RR params and sparse op, and the cases repeat in a shuffled
order so stateful kernels see shape changes.

The Indexer is the target's FP8 policy (Fp8IndexerOp: E4M3 keys with a UE8M0 FP32 scale per token,
group-128 E4M3 queries); the MLA cache and sparse attention are BF16, as on feat/k3.1_dev. The
reference scores use the op's own E4M3 queries and the writer's quantized keys, so it checks the
score, top-k, sparse MLA, V-up and combine stages rather than the quantization.

Run on eight GPUs:
  torchrun --no-python --nproc_per_node=8 bazel-bin/rtp_llm/test/kimi_k31_dsa_dcp_decode_test \
      --tp-size 8 --cases '[[2, 4, 4097], {"queries": 4, "contexts": [4097, 1048576]}]' --repeat 3
A case is [batch, queries, context] or {"queries": Q, "contexts": [...]}. --modes selects the
switch settings of MODES; "dev" is the branch's own path.
"""

import argparse
import faulthandler
import json
import os
import random
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.modules.base.cuda.fp8_indexer_op import Fp8IndexerOp
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_kv_workspace import (
    DsaWorkspaceOwner,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_page_rr_params import (
    DsaPageRRParams,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_page_rr_impl import (
    PageRRSparseMlaOp,
)
from rtp_llm.ops import NcclCommConfig, ParallelismConfig, compute_ops

try:
    from rtp_llm.test import kimi_k31_dsa_fixture as fixture
except ImportError:  # run as a plain script next to the fixture (no Bazel runfiles)
    import kimi_k31_dsa_fixture as fixture

PAGE, KERNEL_PAGE, TOPK = fixture.PAGE, fixture.KERNEL_PAGE, 2048
INDEX_HEADS, INDEX_DIM = 32, fixture.INDEX_DIM
HEADS, LATENT, ROPE, NOPE, V_HEAD = 96, fixture.LATENT, fixture.ROPE, 128, 128
SOFTMAX_SCALE = (NOPE + ROPE) ** -0.5


def parse_case(case):
    if isinstance(case, dict):
        return case["queries"], list(case["contexts"])
    batch, queries, context = case
    return queries, [context] * batch


def case_label(tp, queries, contexts):
    if len(set(contexts)) == 1:
        return f"tp{tp}_b{len(contexts)}_q{queries}_ctx{contexts[0]}"
    return f"tp{tp}_mixed{len(contexts)}_q{queries}_ctx{min(contexts)}-{max(contexts)}"


class DecodeCase:
    """One MTP-verify decode step of one DSA layer on this rank."""

    def __init__(self, parallel, owner, pool, queries, contexts, *, seed, max_seq_len, graph_width):
        tp, rank = parallel.tp_size, parallel.tp_rank
        device = torch.device("cuda")
        self.parallel, self.owner, self.pool = parallel, owner, pool
        self.queries, self.contexts, self.batch = queries, contexts, len(contexts)
        self.local_heads = HEADS // tp
        rows = self.batch * queries
        # History seeds are shared by the ranks of one group (DP groups pass different seeds).
        self.seeds = [seed * 1009 + request for request in range(self.batch)]
        columns = [fixture.owner_columns(c, tp, rank) for c in contexts]
        self.pages = pool.allocate(columns)
        width = (
            fixture.max_owner_columns(max_seq_len, tp) if graph_width
            else max(fixture.group_columns(contexts, tp), 1)
        )
        self.table = torch.zeros(self.batch, width, dtype=torch.int32, device=device)
        for request, pages in enumerate(self.pages):
            self.table[request, : pages.numel()] = pages
        prefixes = [c - queries for c in contexts]
        self.inputs = SimpleNamespace(
            is_prefill=False,
            is_target_verify=queries > 1,  # one query per request is a plain decode
            is_mtp_draft_update=False,
            is_cuda_graph=False,
            input_lengths=torch.full((self.batch,), queries, dtype=torch.int32),
            prefix_lengths=torch.tensor(prefixes, dtype=torch.int32),
            input_lengths_device=torch.full((self.batch,), queries, dtype=torch.int32, device=device),
            cu_seqlens_device=torch.arange(0, rows + 1, queries, dtype=torch.int32, device=device),
            padding_offset=torch.zeros(rows, dtype=torch.int32, device=device),
            logical_token_count=rows,
            physical_token_count=rows,
            prefix_lengths_device=torch.tensor(prefixes, dtype=torch.int32, device=device),
            sequence_lengths_plus_1_device=torch.tensor(contexts, dtype=torch.int32, device=device),
            kv_cache_kernel_block_id_device=self.table,
        )
        attention = SimpleNamespace(tokens_per_block=PAGE, kernel_tokens_per_block=KERNEL_PAGE)
        self.params = DsaPageRRParams(
            # Eager calls: a graph-keyed workspace stays resident per (batch, Q) for the process.
            attention, parallel, max_seq_len=max_seq_len, workspace_owner=owner, is_cuda_graph=False
        )
        self.params.prepare(self.inputs)
        self.params.prepare_execution("indexer")
        self.params.prepare_execution("sfa")
        for request, pages in enumerate(self.pages):
            pool.write_history(self.seeds[request], pages, contexts[request], tp, rank)

        gen = torch.Generator(device=device).manual_seed(seed)

        def randn(*shape, scale):
            return (torch.randn(*shape, generator=gen, device=device) * scale).bfloat16()

        self.index_query = randn(rows, INDEX_HEADS, INDEX_DIM, scale=0.1)
        self.gates = torch.rand(rows, INDEX_HEADS, 1, generator=gen, device=device) * 0.01
        self.mla_query = randn(rows, HEADS, LATENT + ROPE, scale=0.1)
        self.local_query = self.mla_query[
            :, rank * self.local_heads : (rank + 1) * self.local_heads
        ].contiguous()
        self.positions = self.params.positions_d.clone()
        expected = torch.cat([torch.arange(p, p + queries) for p in prefixes]).to(device)
        assert torch.equal(self.positions.long(), expected)
        self.index_cache = SimpleNamespace(kv_cache_base=pool.index)
        self.latent_cache = pool.latent
        self.slot_to_global = pool.slot_to_global([(pages, tp, rank) for pages in self.pages])
        self._references = {}

    def request_of(self, row):
        return row // self.queries

    def write_current_tokens(self, indexer):
        """The verified tokens (procedural like the history) through the production writers."""
        keys, values = [], []
        for request in range(self.batch):
            positions = self.positions[request * self.queries : (request + 1) * self.queries]
            keys.append(fixture.index_keys(self.seeds[request], positions))
            values.append(fixture.latents(self.seeds[request], positions))
        keys, values = torch.cat(keys).bfloat16(), torch.cat(values).bfloat16()
        # The model folds the query scale into the FP32 head weights (dsa_head_weights_fp32).
        self.index_query_fp8, query_scale = indexer.encode_q_k(
            self.index_query, keys, self.index_cache, self.params, self.inputs, self.params
        )
        self.weights = self.gates * query_scale
        compute_ops.concat_and_cache_mla(
            values[:, :LATENT].contiguous(), values[:, LATENT:].contiguous(),
            self.latent_cache, self.params.slot_mapping, "auto", indexer.scale,
        )

    def scores(self, row, chunk=1 << 17):
        """fp32 Indexer scores of a row over its causal global history (cached per request, [Q, length]).

        The FP8 paged MQA: sum over heads of weight * relu(q_e4m3 . k_e4m3) * key scale.
        """
        request = self.request_of(row)
        if request not in self._references:
            first, last = request * self.queries, (request + 1) * self.queries
            positions = self.positions[first:last].long()
            length = int(positions.max()) + 1
            query = self.index_query_fp8[first:last].float()  # [Q, H, D]
            gates = self.weights[first:last, :, 0].float()  # [Q, H]
            score = torch.empty(self.queries, length, device=query.device)
            for start in range(0, length, chunk):
                key_positions = torch.arange(start, min(start + chunk, length), device=query.device)
                keys, key_scale = fixture.fp8_index_keys(self.seeds[request], key_positions)
                logits = torch.relu(torch.einsum("qhd,kd->qhk", query, keys)) * key_scale
                score[:, start : start + key_positions.numel()] = (logits * gates[:, :, None]).sum(1)
            score.masked_fill_(
                torch.arange(length, device=query.device)[None, :] > positions[:, None], -float("inf")
            )
            self._references[request] = score
        return self._references[request][row - request * self.queries]

    def reference(self, row, selected=None):
        """fp32 reference of one row: (top-k positions, scores, this rank's [heads, V])."""
        position = int(self.positions[row])
        score = self.scores(row)[: position + 1]
        order = torch.sort(score, descending=True, stable=True).indices  # ties: lower position
        topk = order[:TOPK]
        chosen = topk if selected is None else selected
        kv = fixture.latents(self.seeds[self.request_of(row)], chosen)
        probs = torch.softmax(self.mla_query[row].float() @ kv.T * SOFTMAX_SCALE, -1)
        output = torch.bmm((probs @ kv[:, :LATENT]).unsqueeze(1), self.owner.value_weights[0].float())
        heads = slice(self.parallel.tp_rank * self.local_heads, (self.parallel.tp_rank + 1) * self.local_heads)
        return topk, score, output.squeeze(1)[heads]


def build_sparse(case):
    sparse = PageRRSparseMlaOp(
        case.local_heads, LATENT, ROPE, NOPE, KERNEL_PAGE, 1.0, TOPK,
        parallelism_config=case.parallel, workspace_owner=case.owner,
    )
    sparse.plan(case.params, case.table, case.inputs)
    return sparse


def build_indexer():
    return Fp8IndexerOp(
        SimpleNamespace(
            indexer_use_fp8=True, index_rope_dim=0, index_num_heads=INDEX_HEADS,
            index_head_dim=INDEX_DIM, index_topk=TOPK,
        )
    )


MODES = {"dev": {}}  # switch settings per mode; each integrated optimization adds its switches
SWITCHES = ()


def set_mode(mode):
    for name in SWITCHES:
        os.environ.pop(name, None)
    os.environ.update(MODES[mode])


def run_layer(case, indexer, sparse):
    params = case.params
    selected = indexer.compute_topk(
        case.index_query_fp8, case.weights, case.index_cache, params, case.inputs, params
    )
    output = sparse.forward(case.local_query, case.latent_cache.view(-1, 1, LATENT + ROPE), selected)
    return selected, output


def check_against_reference(case, selected, output, label):
    """Selection: a valid exact top-k up to fp32 summation-order ties. Output: fp32 on the same keys."""
    rows = case.batch * case.queries
    report = {"case": label, "rows": rows, "exact_sets": 0, "boundary_swaps": 0,
              "max_out_err": 0.0, "failures": []}
    for row in range(rows):
        position = int(case.positions[row])
        chosen = selected[row][selected[row] >= 0].long()
        if (selected[row] == -2).any():
            report["failures"].append((row, "slot outside the row's request"))
            continue
        if chosen.numel() != min(position + 1, TOPK) or not (chosen <= position).all() \
                or chosen.unique().numel() != chosen.numel():
            report["failures"].append((row, f"selection size/causality: {chosen.numel()}"))
            continue
        topk, score, expected = case.reference(row, chosen)
        if torch.equal(chosen.sort().values, topk.sort().values):
            report["exact_sets"] += 1
        else:
            kth = score[topk].min()
            tolerance = 1e-4 * kth.abs() + 1e-6
            if score[chosen].min() < kth - tolerance:
                report["failures"].append((row, "selection is not a top-k"))
                continue
            report["boundary_swaps"] += 1
        error = (output[row].float() - expected).abs()
        report["max_out_err"] = max(report["max_out_err"], error.max().item())
        # Two BF16 roundings (attention output, V-up) cost ~2**-9 of the row's magnitude each,
        # which dominates rows over a handful of keys (their outputs are not averaged down).
        scale = expected.abs().max().item()
        tolerance = 2e-4 + 0.015 * expected.abs() + 4e-3 * scale
        if not torch.isfinite(output[row]).all() or (error > tolerance).any():
            report["failures"].append((row, f"output error {error.max().item():.3g} (row max {scale:.3g})"))
    report["failures"] = report["failures"][:4] + ([f"... {len(report['failures'])} rows"]
                                                    if len(report["failures"]) > 4 else [])
    return report


def init(tp_size):
    world_rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    parallel = ParallelismConfig()
    parallel.tp_size, parallel.dp_size, parallel.ep_size = tp_size, 8 // tp_size, 8
    parallel.world_size = parallel.local_world_size = 8
    parallel.world_rank = parallel.local_rank = world_rank
    parallel.ffn_sp_size = tp_size
    parallel.prefill_cp_config.kv_cache_sharded = True
    nccl = NcclCommConfig()
    nccl.nccl_ip = os.environ["MASTER_ADDR"]
    collective.init_distributed_environment(
        parallel, nccl, int(os.environ["MASTER_PORT"]), disable_custom_all_reduce=True
    )
    return parallel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp-size", type=int, choices=(4, 8), required=True)
    parser.add_argument("--cases", default="[[2, 4, 4097], [4, 4, 65536], [8, 4, 131072]]")
    parser.add_argument("--modes", default='["dev"]')
    parser.add_argument("--repeat", type=int, default=1, help="runs of every case")
    parser.add_argument("--shuffle-seed", type=int, default=None, help="shuffle the run order")
    parser.add_argument("--max-seq-len", type=int, default=1 << 20)
    parser.add_argument("--pool-pages", type=int, default=200_000, help="per-rank cache pages")
    parser.add_argument("--graph-width", action="store_true", help="page tables of max_seq_len width")
    parser.add_argument("--step-timeout", type=int, default=1800, help="seconds before a hung step dumps stacks")
    args = parser.parse_args()
    parallel = init(args.tp_size)
    device = torch.device("cuda")
    # The target's geometry (DsaWorkspaceOwner.from_model): FP8 Indexer, 96 MLA heads.
    owner = DsaWorkspaceOwner(
        index_heads=INDEX_HEADS, index_dim=INDEX_DIM, indexer_use_fp8=True, topk=TOPK, heads=HEADS,
        latent_dim=LATENT, suffix_dim=ROPE, value_dim=V_HEAD, nope_dim=NOPE, tp_size=args.tp_size,
    )
    weights = torch.Generator(device=device).manual_seed(3101)
    owner.value_weights[0] = (
        torch.randn(HEADS, LATENT, V_HEAD, generator=weights, device=device) * 0.05
    ).bfloat16()
    indexer = build_indexer()
    pool = fixture.PagePool(args.pool_pages, device, seed=7 + parallel.world_rank)
    cases = [parse_case(c) for c in json.loads(args.cases)]
    schedule = [(index, run) for run in range(args.repeat) for index in range(len(cases))]
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(schedule)
    modes = json.loads(args.modes)
    failures = 0
    faulthandler.enable(file=sys.stderr, all_threads=True)
    for step, (index, run) in enumerate(schedule):
        queries, contexts = cases[index]
        label = case_label(args.tp_size, queries, contexts)
        if parallel.world_rank == 0:
            print("PROGRESS", step, label, "run", run, flush=True)
        faulthandler.dump_traceback_later(args.step_timeout, exit=True)
        case = DecodeCase(
            parallel, owner, pool, queries, contexts, seed=1000 * index + parallel.dp_rank,
            max_seq_len=args.max_seq_len, graph_width=args.graph_width,
        )
        sparse = build_sparse(case)
        case.write_current_tokens(indexer)
        for mode in modes:
            set_mode(mode)
            selected, output = run_layer(case, indexer, sparse)
            torch.cuda.synchronize()
            dist.barrier()
            report = check_against_reference(case, selected, output, label)
            failed = torch.tensor([len(report["failures"])], device=device)
            dist.all_reduce(failed)
            passed = failed.item() == 0
            failures += not passed
            if parallel.tp_rank == 0 or not passed:
                print(json.dumps({"step": step, "run": run, "world_rank": parallel.world_rank,
                                  "mode": mode, "passed": passed, **report}), flush=True)
        del case, sparse, selected, output
        torch.cuda.empty_cache()
    faulthandler.cancel_dump_traceback_later()
    dist.barrier()
    dist.destroy_process_group()
    if failures:
        raise SystemExit(f"{failures} decode checks failed")


if __name__ == "__main__":
    main()
