import asyncio
import base64
import gzip
import json
import struct
import sys
import tempfile
import threading
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import hook
import zstandard
from bundle import checksum
from configure import configuration
from mckidx_writer import HEADER, REQUEST
from offline import offline_build
from partitioned_index import PartitionedIndex, merge_runs
from prefix_index import SegmentLoss
from verify import source_seal, verify_series


def archive(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def write_segment(root, start, rows, geometries, source=None):
    end = start + 120_000
    shards = root / str(start) / "shards"
    archive(
        shards / "frontend/pod/worker-0-test.jsonl.gz",
        [
            ["header", 4, start, end, f"{start:032x}", "0" * 32],
            *rows,
            ["end", len(rows), 0, 0, *([] if source is None else [source])],
        ],
    )
    archive(
        shards / "vit/pod/vit-test.jsonl.gz",
        [
            ["geometry", 4, start, end, f"{start+1:032x}", "0" * 32],
            *geometries,
            ["end", len(geometries), 0],
        ],
    )
    return {"shards": str(shards), "start_ms": start, "end_ms": end}


def request(
    arrival, keys, *, media=(), observed=None, inline=(), interface="openai_http"
):
    labels = base64.b64encode(bytes().join(bytes([key]) * 16 for key in keys)).decode()
    return [
        "request",
        arrival,
        len(keys) * 128 + 1,
        5,
        16,
        media,
        labels,
        len(keys) // 2,
        128,
        256,
        observed,
        interface,
        inline,
    ]


class DatasetTest(unittest.TestCase):
    def test_partitions_and_media_match_sqlite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b, c = "a" * 32, "b" * 32, "c" * 32
            first = write_segment(
                root,
                1000,
                [
                    request(1001, (1, 2, 3), media=[[a, 0, 128]], observed=9),
                    request(
                        1001, (1, 2, 4), media=[[b, 0, 128]], interface="dashscope_grpc"
                    ),
                    request(
                        1002, (128, 129), media=[[c, 0, 128]], inline=[[c, 64, 64, 128]]
                    ),
                ],
                [["image", a, 96, 128, 128], ["image", b, 96, 128, 128]],
            )
            second = write_segment(
                root,
                121000,
                [
                    request(121001, (1, 2, 3), media=[[a, 0, 128]]),
                    request(
                        121002,
                        (128, 129, 130),
                        media=[[c, 0, 128]],
                        inline=[[c, 64, 64, 128]],
                    ),
                ],
                [["image", a, 96, 128, 128]],
            )
            plan = {
                "segments": [first, second],
                "frontend_workers": {"pod": 1},
                "vit_pods": 1,
                "max_missing_rate": 0,
                "sort_mib": 1,
            }
            baseline = offline_build(plan, root / "sqlite", sqlite=True)["conversion"]
            report = offline_build(plan, root / "sorted")
            actual = report["conversion"]
            for field in (
                "dataset_requests",
                "repeat_tokens_base",
                "repeat_tokens_routing",
                "image_references",
                "replay_media_ids",
            ):
                self.assertEqual(baseline[field], actual[field], field)
            self.assertEqual(report["validation"]["requests_openai_http"], 4)
            self.assertEqual(report["validation"]["requests_dashscope_grpc"], 1)
            self.assertEqual(report["validation"]["observed_outputs"], 1)
            data = (root / "sorted/replay.mckidx").read_bytes()
            magic, count, *_, payload, _, _, _ = HEADER.unpack_from(data)
            self.assertEqual((magic, count), (b"MCKINDEX", 5))
            offset, size, _ = struct.unpack_from(">qii", data, HEADER.size)
            raw = zstandard.ZstdDecompressor().decompress(
                data[payload + offset : payload + offset + size]
            )
            rows = [
                REQUEST.unpack_from(
                    raw, 4 * count + struct.unpack_from(">I", raw, 4 * row)[0]
                )
                for row in range(count)
            ]
            self.assertEqual([row[0] for row in rows], [1, 1, 2, 120001, 120002])
            self.assertEqual([row[2] for row in rows], [9, -1, -1, -1, -1])

    def test_source_counts_merge_across_segments_and_preserve_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            id_a, id_b = "v-71b85312", "v-a2a65be6"
            fixtures = ({id_a: 3}, {id_a: 2, id_b: 1}, None)
            segments = []
            for i, counts in enumerate(fixtures):
                start = 1000 + 120_000 * i
                source = [i + 1, counts] if counts is not None else None
                segments.append(
                    write_segment(root, start, [request(start, (1, 2))], [], source)
                )
            plan = {
                "segments": segments,
                "frontend_workers": {"pod": 1},
                "vit_pods": 1,
                "max_missing_rate": 0,
                "sort_mib": 1,
            }
            expected = {
                "first_frames": 9,
                "missing_service_id": 3,
                "by_service_id": {id_a: 5, id_b: 1},
                "shards_with_source_stats": 2,
            }
            for sqlite in (False, True):
                report = offline_build(plan, root / f"indexed-{sqlite}", sqlite)
                validation = report["validation"]
                self.assertEqual(validation["requests"], 3)
                self.assertEqual(validation["production_source"], expected)
                self.assertEqual(
                    [
                        s["production_source"]["shards_with_source_stats"]
                        for s in validation["segments"]
                    ],
                    [1, 1, 0],
                )
            paths = [
                (
                    Path(s["shards"]) / "frontend",
                    Path(s["shards"]) / "vit",
                    s["start_ms"],
                    s["end_ms"],
                )
                for s in segments
            ]
            report = verify_series(paths, {"pod": 1}, 1, root)
            self.assertEqual(report["production_source"], expected)

            for start in (1000, 121000):
                write_segment(root, start, [request(start, (1, 2))], [])
            legacy = offline_build(plan, root / "legacy")
            self.assertEqual(
                legacy["conversion"]["sha256"],
                checksum(root / "indexed-False/replay.mckidx"),
            )

    def test_invalid_source_counts_fail_validation(self):
        for data in [[-1, {}], [0, {"invalid": 2}], [0, {"v-71b85312": True}]]:
            with self.subTest(data=data), self.assertRaisesRegex(
                ValueError, "invalid source counts"
            ):
                source_seal([data])

    def test_committed_runs_survive_compaction_and_reject_missing_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = PartitionedIndex(root / "index", sort_bytes=1)
            segment = write_segment(
                root, 1000, [request(1001 + n, (1, 2)) for n in range(16)], []
            )
            shards = Path(segment["shards"])
            index.ingest(
                shards / "frontend", shards / "vit", 1000, 121000, {"pod": 1}, 1, 0
            )
            index.close()
            reopened = PartitionedIndex(root / "index")
            paths = [reopened.path / row[0] for row in reopened.state["runs"]]
            self.assertEqual(len(paths), 2)
            self.assertEqual(len(list(merge_runs(paths))), 16)
            bad = write_segment(
                root, 121000, [request(121001, (1, 2), media=[["e" * 32, 0, 128]])], []
            )
            shards = Path(bad["shards"])
            with self.assertRaises(SegmentLoss):
                reopened.ingest(
                    shards / "frontend",
                    shards / "vit",
                    121000,
                    241000,
                    {"pod": 1},
                    1,
                    0,
                )
            self.assertEqual(len(reopened.state["segments"]), 1)
            reopened.close()


class HookTest(unittest.IsolatedAsyncioTestCase):
    async def test_stream_flushes_known_output_and_preserves_request_context(self):
        rows = []
        capture = SimpleNamespace(
            start=1000,
            end=2000,
            arrivals=0,
            lock=threading.Lock(),
            closed=False,
            put=rows.append,
        )
        config = SimpleNamespace(num_return_sequences=1, num_beams=1)
        token = hook._arrival.set((1001, "openai_http"))
        saved = hook._captures
        hook._captures = (capture,)

        async def restore_context():
            hook._arrival.reset(token)
            hook._captures = saved

        self.addAsyncCleanup(restore_context)

        async def backend():
            async def stream():
                state = hook._request.get()
                state["trust_output"] = True
                state["candidate"] = ["request", 1001, 128, 1, 20, [], "", 1, 128, 128]
                yield SimpleNamespace(
                    generate_outputs=[
                        SimpleNamespace(
                            finished=True, aux_info=SimpleNamespace(output_len=7)
                        )
                    ]
                )

            return stream()

        wrapper = hook.after_enqueue(
            SimpleNamespace(
                args=[object(), SimpleNamespace(generate_config=config)],
                kwargs={},
                result=backend(),
            )
        )
        stream = await wrapper
        await anext(stream)
        await asyncio.create_task(stream.aclose())
        self.assertEqual(rows[0][-3:], [7, "openai_http", []])
        self.assertEqual(capture.arrivals, 1)
        self.assertIsNone(hook._request.get())

    async def test_dispatch_rejection_keeps_validated_request_for_replay(self):
        rows = []
        capture = SimpleNamespace(
            start=1000,
            end=2000,
            arrivals=0,
            closed=False,
            key=b"k" * 32,
            lock=threading.Lock(),
            put=rows.append,
        )
        request = SimpleNamespace(
            generate_config=SimpleNamespace(max_new_tokens=20),
            mm_inputs=[],
        )
        visitor = SimpleNamespace(cache_key_seed=0)
        master_client = SimpleNamespace(
            MasterClient=SimpleNamespace(_extract_priority=lambda request: 3),
        )

        async def reject_dispatch():
            hook.after_block_keys(
                SimpleNamespace(args=[[1, 2], 2], kwargs={}, result=[17])
            )
            hook.before_pd(
                SimpleNamespace(
                    kwargs={
                        "input": request,
                        "block_cache_keys": [17],
                        "cache_key_block_size": 2,
                    }
                )
            )
            raise RuntimeError("dispatch rejected")

        async def enqueue():
            async def stream():
                yield await hook.after_route(
                    SimpleNamespace(
                        args=[visitor, request], kwargs={}, result=reject_dispatch()
                    )
                )

            return stream()

        arrival_token = hook._arrival.set((1001, "dashscope_grpc"))
        try:
            with (
                patch.object(hook, "_captures", (capture,)),
                patch.dict(
                    sys.modules, {"rtp_llm.server.master_client": master_client}
                ),
            ):
                stream = await hook.after_enqueue(
                    SimpleNamespace(
                        args=[visitor, request], kwargs={}, result=enqueue()
                    )
                )
                with self.assertRaisesRegex(RuntimeError, "dispatch rejected"):
                    await anext(stream)
        finally:
            hook._arrival.reset(arrival_token)

        self.assertEqual(capture.arrivals, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][:6], ["request", 1001, 2, 3, 20, []])
        self.assertEqual(rows[0][7:], [1, 2, 2, None, "dashscope_grpc", []])
        self.assertIsNone(hook._request.get())

    def test_first_grpc_frame_counts_once_even_without_replay(self):
        capture = SimpleNamespace(
            start=1000,
            end=2000,
            source_ids=Counter(),
            source_missing=0,
            lock=threading.Lock(),
            closed=False,
        )
        saved = hook._captures
        arrival_token = hook._arrival.set(None)
        source_token = hook._source_arrival.set(None)
        hook._captures = (capture,)
        try:

            def mark(ts):
                hook.mark_arrival(
                    SimpleNamespace(
                        args=[SimpleNamespace(start_ts=ts)],
                        hook_config={"config": {"interface": "dashscope_grpc"}},
                    )
                )

            def frame(value):
                request = SimpleNamespace(
                    parameters={
                        "ds_header_attributes": SimpleNamespace(string_param=value)
                    }
                )
                hook.record_source(SimpleNamespace(args=[(), request], kwargs={}))

            mark(1.1)
            frame(
                '{"X-DashScope-Service-ID":"v-71b85312","x-dashscope-requestid":"sample"}'
            )
            frame('{"x-dashscope-service-id":"v-a2a65be6"}')
            mark(1.2)
            frame("{malformed")
            mark(2.0)
            frame('{"x-dashscope-service-id":"v-a2a65be6"}')
            self.assertEqual(capture.source_ids, {"v-71b85312": 1})
            self.assertEqual(capture.source_missing, 1)
        finally:
            hook._captures = saved
            hook._arrival.reset(arrival_token)
            hook._source_arrival.reset(source_token)

    def test_worker_writes_only_source_counters_in_seal(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(threading.Thread, "start"):
                capture = hook.Capture(1000, 2000, b"k" * 32, Path(directory), 0)
            capture.source_ids["v-71b85312"] = 3
            capture.source_missing = 1
            capture.write()
            with gzip.open(capture.path, "rt") as stream:
                header, seal = map(json.loads, stream)
            self.assertEqual(header[:2], ["header", 4])
            self.assertEqual(seal, ["end", 0, 0, 0, [1, {"v-71b85312": 3}]])

    def test_profiles_use_distinct_entry_hooks(self):
        http = configuration("http", "/tmp/shards")["function_hooks"]
        grpc = configuration("grpc", "/tmp/shards")["function_hooks"]
        self.assertEqual(
            Counter(e["config"].get("interface") for e in http)["openai_http"], 2
        )
        self.assertEqual(
            Counter(e["config"].get("interface") for e in grpc)["dashscope_grpc"], 1
        )
        self.assertFalse(any(e.get("before") == "record_source" for e in http))
        self.assertEqual(
            [e["target"] for e in grpc if e.get("before") == "record_source"],
            ["rtp_llm.dash_sc.inference.servicer.extract_span_external_request_id"],
        )

    def test_legacy_profile_hooks_the_monolithic_route(self):
        config = configuration("both", "/tmp/shards", "legacy")
        route_hooks = [
            row for row in config["function_hooks"] if row.get("after") == "after_route"
        ]
        self.assertEqual(len(route_hooks), 1)
        self.assertTrue(route_hooks[0]["target"].endswith(".get_master_route_addrs"))
        self.assertEqual(config["capture"]["runtime"], "legacy")


if __name__ == "__main__":
    unittest.main()
