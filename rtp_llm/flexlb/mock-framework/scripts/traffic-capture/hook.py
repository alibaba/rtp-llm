"""Capture the anonymous prefix keys of requests entering P/D routing."""

import base64
import gzip
import hashlib
import hmac
import json
import logging
import os
import queue
import re
import runpy
import struct
import threading
import time
import uuid
from collections import Counter
from contextvars import ContextVar
from pathlib import Path

_request = ContextVar("frontend_capture_request", default=None)
_arrival = ContextVar("frontend_capture_arrival", default=None)
_source_arrival = ContextVar("frontend_capture_source_arrival", default=None)
_captures = ()
QUEUE = 256
DRAIN_MS = 60_000
KEY = struct.Struct(">Q")
SERVICE_ID = re.compile(r"v-[a-f0-9]{8}")


def digest(key, namespace, data):
    return hmac.digest(key, namespace + data, "sha256")[:16]


class Capture:
    def __init__(self, start, end, key, directory, worker_id):
        self.start, self.end, self.key = start, end, key
        if len(key) != 32 or not 0 < end - start <= 300_000:
            raise ValueError("invalid Frontend capture window")
        shard = uuid.uuid4().hex
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = directory / f"worker-{worker_id}-{os.getpid()}-{shard}.jsonl.gz"
        self.shard = shard
        self.events = queue.Queue(QUEUE)
        self.lock = threading.Lock()
        self.arrivals = self.dropped = self.errors = 0
        self.source_ids = Counter()
        self.source_missing = 0
        self.closed = False
        threading.Thread(
            target=self.write, name="frontend-capture", daemon=True
        ).start()

    def put(self, row):
        with self.lock:
            if self.closed:
                return
            try:
                self.events.put_nowait(row)
            except queue.Full:
                self.dropped += 1

    def error(self):
        with self.lock:
            if not self.closed:
                self.errors += 1

    def write(self):
        partial = self.path.with_suffix(".gz.part")
        try:
            fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as file, gzip.GzipFile(
                fileobj=file, mode="wb", compresslevel=1
            ) as stream:

                def emit(row):
                    stream.write(
                        json.dumps(row, separators=(",", ":")).encode() + b"\n"
                    )

                emit(
                    [
                        "header",
                        4,
                        self.start,
                        self.end,
                        self.shard,
                        digest(self.key, b"check", b"").hex(),
                    ]
                )
                while True:
                    try:
                        row = self.events.get(timeout=0.1)
                    except queue.Empty:
                        with self.lock:
                            if (
                                time.time_ns() // 1_000_000 < self.end + DRAIN_MS
                                or not self.events.empty()
                            ):
                                continue
                            self.closed = True
                            emit(
                                [
                                    "end",
                                    self.arrivals,
                                    self.dropped,
                                    self.errors,
                                    [self.source_missing, self.source_ids],
                                ]
                            )
                            break
                    else:
                        emit(row)
            partial.rename(self.path)
        except Exception as error:
            logging.getLogger(__name__).error(
                "Frontend capture writer failed: %s", type(error).__name__
            )
            with self.lock:
                self.closed = True


def on_worker_start(ctx):
    from rtp_llm import start_server

    controls = runpy.run_path(str(Path(__file__).with_name("bundle.py")))
    key = controls["configured_key"]()
    if key is None:
        return
    root = Path(ctx.config["capture"]["directory"])
    app = ctx.args[0]
    env = app.py_env_configs
    config = env.server_config
    pc = env.parallelism_config
    if ctx.config["capture"].get("runtime") == "legacy":
        # The legacy launcher enumerates local ranks before it assigns the worker's world rank.
        from types import SimpleNamespace

        launch_config = SimpleNamespace(
            parallelism_config=SimpleNamespace(
                world_rank=pc.world_rank - config.rank_id,
                world_size=pc.world_size,
                tp_size=pc.tp_size,
            )
        )
        ranks = list(start_server._iter_serving_ranks(launch_config))
    else:
        ranks = [
            rank % pc.local_world_size
            for rank in start_server._iter_serving_world_ranks(env)
        ]
    workers_per_protocol = len(ranks) * config.frontend_server_count
    worker_id = (
        ranks.index(config.rank_id) * config.frontend_server_count
        + config.frontend_server_id
    )
    workers = workers_per_protocol
    if ctx.config["capture"]["profile"] == "both":
        workers *= 2
        if ctx.hook_config["config"]["protocol"] == "grpc":
            worker_id += workers_per_protocol

    def watch():
        global _captures
        last_start = None
        while True:
            try:
                plan = json.loads((root / "plan.json").read_text())
                start, end = plan["start"], plan["end"]
                if time.time_ns() // 1_000_000 < start and last_start != start:
                    capture = Capture(start, end, key, root / str(start), worker_id)
                    _captures = (
                        capture,
                        *(item for item in _captures if not item.closed),
                    )
                    last_start = start
            except FileNotFoundError:
                pass
            except Exception:
                logging.getLogger(__name__).exception("Frontend capture could not arm")
            time.sleep(0.25)

    if worker_id == 0:
        controls["serve"](root, workers, key)
    threading.Thread(target=watch, name="capture-plan", daemon=True).start()


def mark_arrival(ctx):
    interface = ctx.hook_config["config"]["interface"]
    arrival = (
        int(ctx.args[0].start_ts * 1000)
        if interface == "dashscope_grpc"
        else int(ctx.timestamp * 1000)
    )
    _arrival.set((arrival, interface))
    if interface == "dashscope_grpc":
        _source_arrival.set(arrival)


def record_source(ctx):
    arrival = _source_arrival.get()
    if arrival is None:
        return
    # This request's first frame enters here before generation parameters are parsed.
    _source_arrival.set(None)
    for capture in _captures:
        if capture.start <= arrival < capture.end:
            request = ctx.args[1] if len(ctx.args) > 1 else ctx.kwargs["request"]
            service_id = source_service_id(request)
            with capture.lock:
                if not capture.closed:
                    if service_id is None:
                        capture.source_missing += 1
                    else:
                        capture.source_ids[service_id] += 1
            return


def source_service_id(request):
    parameter = request.parameters.get("ds_header_attributes")
    if parameter is None or not parameter.string_param:
        return None
    try:
        attributes = json.loads(parameter.string_param)
    except (ValueError, RecursionError):
        return None
    if isinstance(attributes, dict):
        for key, value in attributes.items():
            if key.lower() == "x-dashscope-service-id":
                return (
                    value
                    if isinstance(value, str) and SERVICE_ID.fullmatch(value)
                    else None
                )
    return None


def seed_frontend_worker(ctx):
    from rtp_llm.server.cache_key_routing import get_v41_cache_key_seed

    worker = ctx.args[0]
    model_config = ctx.args[2] if len(ctx.args) > 2 else ctx.kwargs["model_config"]
    worker.backend_rpc_server_visitor.cache_key_seed = get_v41_cache_key_seed(
        model_config
    )


def request_state(config):
    arrival = _arrival.get()
    if arrival is None:
        return None
    timestamp, interface = arrival
    for capture in _captures:
        if capture.start <= timestamp < capture.end:
            with capture.lock:
                if capture.closed:
                    return None
                capture.arrivals += 1
            return {
                "capture": capture,
                "arrival": timestamp,
                "interface": interface,
                "trust_output": os.environ.get("MOCK_CAPTURE_REAL_OUTPUT") == "1"
                and getattr(config, "num_return_sequences", 0) <= 1
                and getattr(config, "num_beams", 1) <= 1
                and not getattr(config, "variable_num_beams", ()),
            }
    return None


def finish(state):
    candidate = state.get("candidate")
    observed = state.get("observed")
    geometry = state.get("inline_geometry", [])
    if candidate is None:
        prepared = state.get("prepared")
        if prepared is None:
            return
        candidate, geometry = prepared
        observed = None
    state["capture"].put([*candidate, observed, state["interface"], geometry])


def observe(state, output):
    if not state["trust_output"]:
        return
    rows = getattr(output, "generate_outputs", ())
    if len(rows) == 1 and rows[0].finished:
        count = getattr(rows[0].aux_info, "output_len", None)
        candidate = state.get("candidate")
        if (
            candidate is not None
            and type(count) is int
            and count > 0
            and (candidate[4] is None or count <= candidate[4])
        ):
            state["observed"] = count


def after_enqueue(ctx):
    if not _captures or _arrival.get() is None:
        return
    request = ctx.args[1] if len(ctx.args) > 1 else ctx.kwargs["input"]
    state = request_state(request.generate_config)
    if state is None:
        return
    original = ctx.result

    async def start():
        try:
            outputs = await original
        except BaseException:
            finish(state)
            raise

        async def stream():
            try:
                while True:
                    token = _request.set(state)
                    try:
                        output = await anext(outputs)
                    except StopAsyncIteration:
                        break
                    finally:
                        _request.reset(token)
                    observe(state, output)
                    yield output
            finally:
                try:
                    await outputs.aclose()
                finally:
                    finish(state)

        return stream()

    return start()


def after_batch(ctx):
    if not _captures or _arrival.get() is None:
        return
    inputs = ctx.args[1] if len(ctx.args) > 1 else ctx.kwargs["inputs"]
    original = ctx.result
    states = {id(item): request_state(item.generate_config) for item in inputs}

    async def collect():
        token = _request.set(states)
        try:
            outputs = await original
            for item, output in zip(inputs, outputs):
                state = states[id(item)]
                if state is not None:
                    observe(state, output)
            return outputs
        finally:
            _request.reset(token)
            for state in states.values():
                if state is not None:
                    finish(state)

    return collect()


def after_route(ctx):
    active = _request.get()
    if active is None:
        return
    visitor = ctx.args[0]
    request = ctx.args[1] if len(ctx.args) > 1 else ctx.kwargs["input"]
    state = active if "capture" in active else active.get(id(request))
    if state is None:
        return
    original = ctx.result

    async def route():
        token = _request.set(state)
        try:
            state.pop("candidate", None)
            state.pop("observed", None)
            state.pop("inline_geometry", None)
            state.pop("physical_keys", None)
            state.pop("expanded", None)
            state["request"] = request
            state["visitor"] = visitor
            result = await original
            if result is not None:
                state.pop("candidate", None)
            return result
        except BaseException:
            state.pop("candidate", None)
            raise
        finally:
            state.pop("request", None)
            state.pop("visitor", None)
            state.pop("physical_keys", None)
            state.pop("expanded", None)
            _request.reset(token)

    return route()


def after_expand(ctx):
    state = _request.get()
    if state is None or "request" not in state:
        return
    state.pop("expanded", None)
    if ctx.result[1] is None:
        return
    keys = ctx.args[3]
    spans = ctx.kwargs.get("expanded_spans")
    if spans is not None and len(keys) == len(spans):
        state["expanded"] = list(zip(keys, spans))


def after_block_keys(ctx):
    state = _request.get()
    if state is not None and "request" in state:
        seed = ctx.kwargs.get("cache_key_seed", ctx.args[3] if len(ctx.args) > 3 else 0)
        expected_seed = state["visitor"].cache_key_seed
        state["physical_keys"] = ctx.result if seed == expected_seed else None
        state["block_size"] = ctx.args[1]
        state["expanded_length"] = len(ctx.args[0])


def media_layout(state, request):
    key = state["capture"].key
    removed = 0
    media, geometry = [], []
    v41 = getattr(request, "v41_inputs", None)
    if v41 is not None and v41.images:
        for image in v41.images:
            if not image.content_sha256 or not image.processor_identity:
                raise ValueError("missing prepared image identity")
            identity = digest(
                key,
                b"v41-image",
                (image.content_sha256 + image.processor_identity).encode(),
            ).hex()
            width = image.n_vit_w * int(image.patches.shape[-1])
            height = image.n_vit_h * int(image.patches.shape[-2])
            count = int(image.types.numel())
            media.append([identity, image.start - removed, count])
            geometry.append([identity, width, height, count])
            removed += count
        return media, geometry
    if not request.mm_inputs:
        return media, geometry
    expanded = state.get("expanded")
    if expanded is None or len(expanded) != len(request.mm_inputs):
        return None, geometry
    for identity, (offset, count) in expanded:
        media.append(
            [digest(key, b"image", identity.encode()).hex(), offset - removed, count]
        )
        removed += count
    return media, geometry


def before_pd(ctx):
    state = _request.get()
    args = ctx.kwargs
    if state is None or "request" not in state or args.get("vit_only"):
        return
    request = state["request"]
    if args.get("input") is not request:
        return
    state.pop("candidate", None)
    physical = state.get("physical_keys")
    if physical is None:
        return
    try:
        from rtp_llm.server.master_client import MasterClient

        media, geometry = media_layout(state, request)
        if media is None:
            return
        block, size = state["block_size"], args["cache_key_block_size"]
        length = state["expanded_length"]
        maximum = request.generate_config.max_new_tokens
        if maximum == 0:
            return
        priority = MasterClient._extract_priority(request)
        if (
            type(block) is not int
            or block <= 0
            or type(size) is not int
            or size < block
            or size % block
        ):
            raise ValueError("invalid cache key block sizes")
        stride = size // block
        route = args["block_cache_keys"]
        if (
            len(physical) != length // block
            or route != physical[stride - 1 :: stride]
            or maximum is not None
            and (type(maximum) is not int or maximum < 0)
            or priority < 0
        ):
            raise ValueError("capture disagrees with P/D routing")
        keys = b"".join(
            digest(state["capture"].key, b"block", KEY.pack(key % (1 << 64)))
            for key in physical
        )
        state["candidate"] = [
            "request",
            state["arrival"],
            length,
            priority,
            maximum,
            media,
            base64.b64encode(keys).decode("ascii"),
            len(route),
            block,
            size,
        ]
        state["inline_geometry"] = geometry
        # Preserve a validated arrival when scheduling or dispatch fails.
        state["prepared"] = (state["candidate"], geometry)
    except Exception:
        state["capture"].error()
