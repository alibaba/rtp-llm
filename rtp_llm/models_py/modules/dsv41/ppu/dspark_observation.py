"""Opt-in, bounded native DSpark diagnostics; never a cache transport.

The normal model path does not import this module. Enabled correctness runs
copy selected physical pages to host and perform synchronous task-local I/O;
they provide no performance evidence. Limits, errors and unfinished captures
are explicit and are rejected by the independent offline test reader.
"""

import atexit
import functools
import hashlib
import json
import logging
import os
import threading
import time
from collections import defaultdict
from itertools import count
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
FULL = {"global_kv_1", "global_kv_2", "indexer_kv"}
_INSTANCES = count()


def diagnostic(method):
    @functools.wraps(method)
    def call(self, *args, **kwargs):
        if self.closed or self.failed:
            return None
        try:
            return method(self, *args, **kwargs)
        except Exception as error:
            # Diagnostics cannot turn a sampler result or a stop vote into a
            # different result. Missing/error evidence fails the offline gate.
            self.failed = True
            logger.exception("PPU DSpark diagnostic failed in %s", method.__name__)
            try:
                self.event("error", operation=method.__name__, reason=repr(error))
            except Exception:
                pass
            return None

    return call


def request_ids(inputs, count):
    value = getattr(inputs, "request_id", None)
    if isinstance(value, torch.Tensor) and value.numel() == count:
        return [int(x) for x in value.detach().cpu().reshape(-1).tolist()]
    return None


def full_pool_neighbor_positions(
    tables, span, entries, ratio, block_count, prefixes, lengths, kind, selected
):
    """Allocated logical neighbors outside every potential FULL write address.

    Verify uses the union of all candidate-version writes. A selected accepted
    version can only write a subset. Prefill exports the entire valid prefix,
    even when the capture itself samples only its tail pages.
    """
    ranges = [
        (
            (
                0 if kind == "prefill" else base // ratio * ratio,
                (base + length) // ratio * ratio,
            )
            if kind in ("prefill", "verify")
            else (0, 0)
        )
        for base, length in zip(prefixes, lengths)
    ]

    def address(row, position):
        if position < 0 or position // span >= len(tables[row]):
            return None
        block = int(tables[row][position // span])
        offset = (position % span) // ratio
        if not 0 <= block < block_count or offset >= entries:
            return None
        return block, offset

    def can_write(candidate):
        block, offset = candidate
        for row, (start, stop) in enumerate(ranges):
            if stop <= start:
                continue
            for logical in range(start // span, (stop - 1) // span + 1):
                if logical >= len(tables[row]):
                    raise ValueError("FULL write range exceeds native table")
                if tables[row][logical] != block:
                    continue
                lower = logical * span + offset * ratio
                upper = min((logical + 1) * span, lower + ratio)
                position = (lower + ratio - 1) // ratio * ratio
                if position < upper and start <= position < stop:
                    return True
        return False

    probes = []
    for row, (start, stop) in enumerate(ranges):
        if stop > start:
            candidates = (start - ratio, stop)
        else:
            values = selected[row] or [max(0, prefixes[row] - 1)]
            candidates = (
                min(values) // ratio * ratio - ratio,
                (max(values) // ratio + 1) * ratio,
            )
        for position in candidates:
            candidate = address(row, position)
            if candidate is not None and not can_write(candidate):
                probes.append([row, position])
    return probes


def noop_csa_witness_positions(table, span, block_count, end):
    """Observe owned pages, never synthesize a CSA import/update position."""
    if span <= 0 or end < 0:
        raise ValueError("invalid no-op native request geometry")
    logical_end = (end + span - 1) // span
    if logical_end > len(table):
        raise ValueError("no-op native request exceeds table")
    witnesses = []
    for logical, physical in enumerate(table[:logical_end]):
        if physical in (-1, 0):  # NULL holes and reserved padding own no request bytes.
            continue
        if not 0 <= physical < block_count:
            raise ValueError("no-op native table has an invalid physical block")
        witnesses.append(min((logical + 1) * span, end) - 1)
    return witnesses


class NativeDSparkObserver:
    input_request_ids = staticmethod(request_ids)

    def __init__(self, runner, directory):
        self.model = getattr(runner.core, "dspark_model_role", "target")
        self.directory = Path(directory).resolve()
        task_directory = Path(os.environ["RTP_V41_DSPARK_OUTPUT"]).resolve()
        if not self.directory.is_relative_to(task_directory):
            raise ValueError(
                "native diagnostic sink must stay in its task output directory"
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        self.request_limit = int(
            os.environ.get("RTP_V41_DSPARK_OBSERVE_REQUEST_LIMIT", "128")
        )
        self.capture_limit = int(
            os.environ.get("RTP_V41_DSPARK_OBSERVE_CAPTURE_LIMIT", "512")
        )
        self.byte_limit = int(
            os.environ.get("RTP_V41_DSPARK_OBSERVE_BYTE_LIMIT", str(2 * 1024**3))
        )
        self.block_limit = int(
            os.environ.get("RTP_V41_DSPARK_OBSERVE_BLOCK_LIMIT", "32")
        )
        self.vote_limit = int(
            os.environ.get("RTP_V41_DSPARK_OBSERVE_VOTE_LIMIT", "65536")
        )
        self.frame_limit = int(
            os.environ.get("RTP_V41_DSPARK_OBSERVE_FRAME_LIMIT", "65536")
        )
        self.selected_rounds = {
            int(x)
            for x in os.environ.get("RTP_V41_DSPARK_OBSERVE_ROUNDS", "0,1,2,3").split(
                ","
            )
        }
        if (
            min(
                self.request_limit,
                self.capture_limit,
                self.byte_limit,
                self.block_limit,
                self.vote_limit,
                self.frame_limit,
            )
            < 1
            or not self.selected_rounds
            or min(self.selected_rounds) < 0
        ):
            raise ValueError("invalid native diagnostic bounds")
        source = Path(os.environ["RTP_V41_TEST_SOURCE_ROOT"])
        runtime = Path(os.environ["RTP_V41_TEST_RUNTIME"])
        self.source = json.loads((source / ".v41_snapshot.json").read_text())
        self.runtime = json.loads((runtime / ".v41_snapshot.json").read_text())
        for key in ("snapshot_id", "source_sha256", "public_commit", "internal_commit"):
            if self.source[key] != self.runtime[key]:
                raise ValueError(f"diagnostic source/runtime mismatch {key}")
        self.pid = os.getpid()
        self.instance = next(_INSTANCES)
        self.stem = f"observer-{self.model}-dp{runner.dp_rank}-cp{runner.cp_rank}-p{self.pid}-m{self.instance}"
        self.file = (self.directory / (self.stem + ".jsonl")).open("x")
        self.event_lock = threading.RLock()
        self.sequence = self.frames = self.vote_index = self.votes = 0
        self.last_vote_index = -1
        self.capture_count = self.bytes_written = 0
        self.closed = self.failed = False
        self.last_idle = False
        self.warmup = self.last_vote_warmup = False
        self.seen = defaultdict(set)
        self.rounds = defaultdict(int)
        self.open = {}
        self.event(
            "start",
            schema_version=1,
            model=self.model,
            pid=self.pid,
            instance=self.instance,
            dp_rank=runner.dp_rank,
            dp_size=runner.dp_size,
            cp_rank=runner.cp_rank,
            cp_size=runner.cp_size,
            source={
                k: self.source[k]
                for k in (
                    "snapshot_id",
                    "source_sha256",
                    "public_commit",
                    "internal_commit",
                )
            },
            runtime={
                k: self.runtime[k]
                for k in (
                    "snapshot_id",
                    "source_sha256",
                    "public_commit",
                    "internal_commit",
                )
            },
            window=runner.core.geometry.window,
            device=str(runner.core.device),
            hardware=(
                torch.cuda.get_device_name(runner.core.device)
                if torch.device(runner.core.device).type == "cuda"
                else "cpu"
            ),
            limits=dict(
                requests=self.request_limit,
                captures=self.capture_limit,
                bytes=self.byte_limit,
                blocks_per_pool=self.block_limit,
                votes=self.vote_limit,
                frames=self.frame_limit,
                rounds=sorted(self.selected_rounds),
            ),
            selection="First bounded scheduler IDs per kind; if proposal ABI has no IDs, bounded actual native allocation/table views are explicitly marked unidentified. Only declared rounds are captured.",
            cost="Enabled diagnostics synchronize selected copies and write host artifacts; no performance evidence",
            idle_compression="Only consecutive identical all-empty votes are omitted; indices count every existing reduction, and the first post-work empty transition is retained",
        )
        atexit.register(self.close)

    def event(self, event, **data):
        with self.event_lock:
            row = dict(
                event=event,
                sequence=self.sequence,
                monotonic_ns=time.monotonic_ns(),
                **data,
            )
            self.file.write(json.dumps(row, sort_keys=True) + "\n")
            self.file.flush()
            self.sequence += 1

    def limit(self, name, actual, maximum):
        self.event("limit", name=name, actual=actual, maximum=maximum)
        self.failed = True

    @diagnostic
    def vote(self, local_rows, global_rows, local_stop, global_stop, warmup=False):
        index = self.vote_index
        self.vote_index += 1
        idle = global_rows == 0 and not global_stop
        if idle and self.last_idle and warmup == self.last_vote_warmup:
            return
        self.last_idle = idle
        self.last_vote_warmup = warmup
        if self.votes >= self.vote_limit:
            self.limit("votes", self.votes + 1, self.vote_limit)
            return
        self.event(
            "vote",
            vote_index=index,
            omitted_idle_votes=index - self.last_vote_index - 1,
            local_rows=int(local_rows),
            global_rows=int(global_rows),
            local_stop=bool(local_stop),
            global_stop=bool(global_stop),
            warmup=bool(warmup),
        )
        self.last_vote_index = index
        self.votes += 1

    @staticmethod
    def valid_positions(key, end, window, ratio, sample=False):
        if key[1] in FULL:
            count = end // ratio
            return range(max(0, count - 2) if sample else 0, count)
        if key[1] == "swa_kv":
            return range(max(0, end - window), end)
        return [end - 1] if end % 2 else []

    def _specs(self, bridge, prefixes, lengths, kind, positions):
        specs = []
        window = bridge.logical.geometry.window
        for key, pool in bridge.pools.items():
            ratio = bridge.logical.geometry.ratio(key[0])
            tables = [
                pool.table[row].detach().cpu().clone().tolist()
                for row in range(len(prefixes))
            ]
            blocks, neighbors = set(), []
            selected_rows, semantic_rows, noop_rows = [], [], []
            for row, (base, length) in enumerate(zip(prefixes, lengths)):
                end = base + length
                if positions is not None:
                    affected = list(positions[row])
                elif kind == "prefill":
                    affected = list(
                        self.valid_positions(key, end, window, ratio, sample=True)
                    )
                    if key[1] in FULL:
                        affected = [x * ratio for x in affected]
                elif key[1] in FULL:
                    affected = list(
                        range(base // ratio * ratio, end // ratio * ratio, ratio)
                    )
                elif key[1] == "swa_kv":
                    affected = list(range(max(0, base - window), end))
                else:
                    affected = [n - 1 for n in range(base + 1, end + 1) if n % 2]
                imported = list(
                    self.valid_positions(key, base, window, ratio, sample=True)
                )
                if key[1] in FULL:
                    imported = [x * ratio for x in imported]
                selected = affected + imported
                semantic_rows.append(set(selected))
                noop = key[1] == "csa_state" and not selected
                if noop:
                    selected = noop_csa_witness_positions(
                        tables[row], pool.span, pool.tensor.shape[0], end
                    )
                    noop_rows.append(
                        dict(
                            row=row,
                            end=end,
                            witness_positions=list(selected),
                            coverage=(
                                "UNTOUCHED_OWNED_PAGES"
                                if selected
                                else "NO_OWNED_PAGES_NO_BYTE_COVERAGE"
                            ),
                        )
                    )
                elif not selected:
                    selected = [max(0, base - 1)]
                if key[1] not in FULL and selected:
                    candidates = [max(0, min(selected) - 1), max(selected) + 1]
                    for position in candidates:
                        if noop and position >= end:
                            continue  # Future/padding cells are not request-owned.
                        logical = position // pool.span
                        if (
                            logical < len(tables[row])
                            and 0 <= tables[row][logical] < pool.tensor.shape[0]
                            and (not noop or tables[row][logical] > 0)
                        ):
                            selected.append(position)
                            neighbors.append([row, position])
                selected_rows.append(selected)
            if key[1] in FULL:
                neighbors = full_pool_neighbor_positions(
                    tables,
                    pool.span,
                    pool.entries,
                    ratio,
                    pool.tensor.shape[0],
                    prefixes,
                    lengths,
                    kind,
                    selected_rows,
                )
                for row, position in neighbors:
                    selected_rows[row].append(position)
            for row, selected in enumerate(selected_rows):
                for position in selected:
                    logical = position // pool.span
                    if logical >= len(tables[row]):
                        raise ValueError("selected native row exceeds table")
                    physical = int(tables[row][logical])
                    unallocated = not 0 <= physical < pool.tensor.shape[0]
                    if key[1] == "csa_state" and position in semantic_rows[row]:
                        unallocated = not 0 < physical < pool.tensor.shape[0]
                    if unallocated:
                        raise ValueError("selected native row is unallocated")
                    blocks.add(physical)
            if len(blocks) > self.block_limit:
                raise ValueError("selected physical block limit exceeded")
            spec = dict(
                key=key,
                tag=key[1],
                framework_tag=pool.framework_tag,
                entries=pool.entries,
                span=pool.span,
                ratio=ratio,
                tables=tables,
                blocks=sorted(blocks),
                shape=list(pool.tensor.shape),
                dtype=str(pool.tensor.dtype),
                data_ptr=pool.tensor.data_ptr(),
                device=str(pool.tensor.device),
                neighbor_positions=neighbors,
                full_neighbor_scan_complete=key[1] in FULL,
                native_pool=pool,
            )
            if key[1] == "csa_state":
                spec["noop_rows"] = noop_rows
            specs.append(spec)
        return specs

    @staticmethod
    def _snapshot(specs):
        return [
            spec["native_pool"]
            .tensor.index_select(
                0,
                torch.tensor(
                    spec["blocks"],
                    dtype=torch.int64,
                    device=spec["native_pool"].tensor.device,
                ),
            )
            .contiguous()
            .view(torch.uint8)
            .detach()
            .cpu()
            .clone()
            for spec in specs
        ]

    def _cache_payloads(self, bridge, cache, specs, ends, sample=False):
        result = []
        for spec in specs:
            key, ratio = spec["key"], spec["ratio"]
            for row, end in enumerate(ends):
                valid = list(
                    self.valid_positions(
                        key, end, bridge.logical.geometry.window, ratio, sample
                    )
                )
                if key[1] in FULL:
                    valid = [x * ratio for x in valid]
                valid = [
                    p
                    for p in valid
                    if spec["tables"][row][p // spec["span"]] in spec["blocks"]
                ]
                if not valid:
                    continue
                # Logical block tables/slots are CPU metadata. _gather moves
                # selected indices to its pool device; do not index this CPU
                # slot vector with a device tensor.
                positions = torch.tensor(valid, dtype=torch.int64)
                if key[1] in FULL:
                    all_slots = cache._slots_for_blocks(key, row, end // ratio)
                    slots = all_slots[positions // ratio]
                else:
                    slots = cache.slots_at_positions(*key, row, positions)
                result.append(
                    dict(
                        key=key,
                        row=row,
                        positions=positions.cpu().clone(),
                        packed=cache._gather(key, slots).detach().cpu().clone(),
                    )
                )
        return result

    @diagnostic
    def begin(
        self, kind, bridge, ids, prefixes, lengths, *, cache=None, positions=None
    ):
        if self.frames >= self.frame_limit:
            self.limit("frames", self.frames + 1, self.frame_limit)
            return None
        prefixes, lengths = [int(x) for x in prefixes], [int(x) for x in lengths]
        if ids is not None and len(ids) != len(prefixes):
            raise ValueError("observer request identity/row mismatch")
        first_pool = next(iter(bridge.pools.values()))
        identities = (
            ids
            if ids is not None
            else [
                (
                    first_pool.tensor.data_ptr(),
                    tuple(first_pool.table[row].detach().cpu().tolist()),
                )
                for row in range(len(prefixes))
            ]
        )
        rounds = []
        selected_requests = True
        for identity in identities:
            key = (kind, identity)
            if (
                identity not in self.seen[kind]
                and len(self.seen[kind]) < self.request_limit
            ):
                self.seen[kind].add(identity)
            if identity in self.seen[kind]:
                rounds.append(self.rounds[key])
                self.rounds[key] += 1
            else:
                rounds.append(-1)
                selected_requests = False
        selected = selected_requests and all(r in self.selected_rounds for r in rounds)
        frame = self.frames
        self.frames += 1
        state = dict(
            schema_version=1,
            model=self.model,
            kind=kind,
            capture_id=f"{self.stem}-{frame}",
            request_ids=ids,
            prefixes=prefixes,
            lengths=lengths,
            rounds=rounds,
            window=bridge.logical.geometry.window,
            began_ns=time.monotonic_ns(),
            selected=selected,
            expected=[],
            imports=[],
            observation_positions=(
                [[int(p) for p in row] for row in positions]
                if positions is not None
                else None
            ),
            bridge=bridge,
        )
        self.event(
            "begin",
            **{
                k: state[k]
                for k in (
                    "kind",
                    "capture_id",
                    "request_ids",
                    "prefixes",
                    "lengths",
                    "rounds",
                    "began_ns",
                    "selected",
                )
            },
        )
        self.open[state["capture_id"]] = state
        if not selected:
            return state
        if (
            self.capture_count + sum(s["selected"] for s in self.open.values())
            > self.capture_limit
        ):
            self.limit("captures", self.capture_count + 1, self.capture_limit)
            return state
        specs = self._specs(bridge, prefixes, lengths, kind, positions)
        estimated = (
            sum(
                len(s["blocks"])
                * s["shape"][1]
                * s["native_pool"].tensor.element_size()
                for s in specs
            )
            * 6
            + len(specs) * 65536
        )
        if self.bytes_written + estimated > self.byte_limit:
            self.limit("bytes", self.bytes_written + estimated, self.byte_limit)
            return state
        state.update(
            specs=specs, before=self._snapshot(specs), estimated_bytes=estimated
        )
        if cache is not None:
            state["imports"] = self._cache_payloads(
                bridge, cache, specs, prefixes, sample=True
            )
        return state

    @diagnostic
    def compute_stage(self, state, rows, positions, tokens):
        if state is None:
            return
        stage = state.get("compute_stages", 0)
        state["compute_stages"] = stage + 1
        self.event(
            "compute",
            kind=state["kind"],
            capture_id=state["capture_id"],
            stage=stage,
            request_ids=[state["request_ids"][row] for row in rows],
            positions=[int(x) for x in positions],
            tokens=[int(x) for x in tokens],
        )

    @diagnostic
    def verified(self, state):
        if state is None:
            return
        self.event("verified", kind=state["kind"], capture_id=state["capture_id"])
        if state["selected"]:
            state["verified"] = self._snapshot(state["specs"])

    @diagnostic
    def add_payload(self, state, key, row, positions, packed):
        if state is not None and state["selected"] and positions.numel():
            state["expected"].append(
                dict(
                    key=key,
                    row=row,
                    positions=positions.detach().cpu().clone(),
                    packed=packed.detach().cpu().clone(),
                )
            )

    @diagnostic
    def finish(self, state, outcome, accepted, *, versions=None, cache=None):
        if state is None:
            return
        accepted = [int(x) for x in accepted]
        self.event(
            "finish",
            kind=state["kind"],
            capture_id=state["capture_id"],
            request_ids=state["request_ids"],
            prefixes=state["prefixes"],
            lengths=state["lengths"],
            rounds=state["rounds"],
            accepted_input_rows=accepted,
            outcome=outcome,
        )
        if state["selected"]:
            if versions is not None:
                state["candidate_ends"] = [
                    [int(version.end) for version in row] for row in versions
                ]
                state["selected_versions"] = []
                for row, count in enumerate(accepted):
                    if count:
                        version = versions[row][count - 1]
                        state["selected_versions"].append(
                            dict(
                                row=row, end=int(version.end), accepted_input_rows=count
                            )
                        )
                        for key, positions, packed in version.payloads:
                            if positions.numel():
                                state["expected"].append(
                                    dict(
                                        key=key,
                                        row=row,
                                        positions=positions.detach().cpu().clone(),
                                        packed=packed.detach().cpu().clone(),
                                    )
                                )
            if cache is not None:
                ends = [p + n for p, n in zip(state["prefixes"], accepted)]
                state["expected"] = self._cache_payloads(
                    state["bridge"], cache, state["specs"], ends
                )
            if outcome in ("abort", "rollback", "proposal"):
                state["expected"] = []
            after = self._snapshot(state["specs"])
            pools = []
            for index, spec in enumerate(state["specs"]):
                pool = {k: v for k, v in spec.items() if k != "native_pool"}
                pool.update(before=state["before"][index], after=after[index])
                if state["kind"] in ("verify", "proposal"):
                    pool["verified"] = state.get("verified", after)[index]
                pools.append(pool)
            payload = {
                k: v
                for k, v in state.items()
                if k not in ("bridge", "specs", "before", "verified", "estimated_bytes")
            }
            payload.update(
                complete=True,
                pools=pools,
                pool_keys=[s["key"] for s in state["specs"]],
                outcome=outcome,
                accepted_input_rows=accepted,
            )
            path = self.directory / (state["capture_id"] + ".pt")
            torch.save(payload, path)
            size = path.stat().st_size
            if self.bytes_written + size > self.byte_limit:
                self.limit(
                    "serialized_bytes", self.bytes_written + size, self.byte_limit
                )
            self.bytes_written += size
            self.capture_count += 1
            self.event(
                "capture",
                kind=state["kind"],
                capture_id=state["capture_id"],
                file=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bytes=size,
            )
        self.open.pop(state["capture_id"])

    def close(self):
        if self.closed:
            return
        try:
            self.event(
                "complete",
                open_captures=len(self.open),
                capture_count=self.capture_count,
                bytes=self.bytes_written,
                failed=self.failed,
                vote_count=self.votes,
                total_existing_votes=self.vote_index,
                total_frames=self.frames,
            )
            self.file.close()
        except Exception:
            logger.exception("PPU DSpark diagnostic close failed")
        finally:
            self.closed = True


def create_observer(runner, directory):
    try:
        return NativeDSparkObserver(runner, directory)
    except Exception:
        logger.exception("PPU DSpark diagnostic initialization failed")
        return None
