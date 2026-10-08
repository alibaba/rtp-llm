import unittest

from analyze_modeling_graph_trace import analyze


def event(name, ts, dur, cat="cpu_op", pid=10, tid=20, **args):
    return dict(ph="X", name=name, ts=ts, dur=dur, cat=cat, pid=pid, tid=tid, args=args)


def round_events():
    data = [event("executor.mtp.decode_step(decode_stream_size=32)", 0, 1000)]
    # Very expensive framework GPU work must not enter either modeling metric.
    data.append(
        event(
            "sampler_and_bookkeeping",
            1,
            999,
            cat="kernel",
            pid=0,
            tid=0,
            correlation=999,
        )
    )
    for i, (width, span) in enumerate(((1, 5), (1, 7), (4, 30), (4, 11))):
        prefill = i == 3
        ts = 100 + i * 100
        label = f'cuda_graph.forward(replay{"Prefill" if prefill else "Decode"},B=32,capture=32,Q={width},T={32*width},fake=0)'
        data.extend(
            (
                event(label, ts, 10),
                event("cudaGraphLaunch", ts + 1, 1, cat="cuda_runtime", correlation=i),
                event(
                    "matmul",
                    ts + 20,
                    span - 2,
                    cat="kernel",
                    pid=0,
                    tid=0,
                    correlation=i,
                ),
                event(
                    "model_collective",
                    ts + span + 18,
                    2,
                    cat="kernel",
                    pid=0,
                    tid=1,
                    correlation=i,
                ),
            )
        )
    return data


class ModelingBoundaryTest(unittest.TestCase):
    def test_correlated_graph_gpu_spans_exclude_framework_and_include_communications(
        self,
    ):
        result = analyze(round_events())
        self.assertEqual(len(result["complete_rounds"]), 1)
        record = result["complete_rounds"][0]
        self.assertEqual(record["target_verify_gpu_us"], 30)
        self.assertEqual(record["mtp_modeling_gpu_us"], 23)
        self.assertEqual([p["gpu_span_us"] for p in record["proposal"]], [5, 7])
        self.assertEqual(record["update"][0]["gpu_span_us"], 11)

    def test_missing_update_and_eager_cannot_pass(self):
        for data in (
            round_events()[:-4],
            round_events() + [event("py_model.forward(normal)", 500, 10)],
        ):
            self.assertEqual(analyze(data)["complete_rounds"], [])

    def test_unknown_historical_update_shape_cannot_pass(self):
        data = round_events()
        for e in data:
            if e["name"].startswith("cuda_graph.forward(replayPrefill"):
                e["name"] = "cuda_graph.forward(replayPrefill)"
        self.assertEqual(analyze(data)["complete_rounds"], [])
        data.append(
            event(
                "cuda_graph.modeling(role=3,logical_b=32,physical_b=32,q=4,bucket=128,capture_b=32,capture_t=128)",
                400,
                10,
            )
        )
        self.assertEqual(len(analyze(data)["complete_rounds"]), 1)

    def test_larger_capture_capacity_cannot_impersonate_actual_batch(self):
        data = round_events()
        for e in data:
            if e["name"].startswith("cuda_graph.forward(replayPrefill"):
                e["name"] = "cuda_graph.forward(replayPrefill)"
        data.append(
            event(
                "cuda_graph.modeling(role=3,logical_b=32,physical_b=32,q=4,bucket=128,capture_b=64,capture_t=256)",
                400,
                10,
            )
        )
        self.assertEqual(analyze(data)["complete_rounds"], [])
        data = round_events()
        for e in data:
            e["name"] = e["name"].replace("capture=32", "capture=64")
        self.assertEqual(analyze(data)["complete_rounds"], [])

    def test_partial_gpu_window_cannot_pass(self):
        data = [
            e
            for e in round_events()
            if not (e["cat"] == "kernel" and e["args"].get("correlation") == 3)
        ]
        result = analyze(data)
        self.assertEqual(result["complete_rounds"], [])
        self.assertEqual(result["graph_launches_missing_gpu_activities"], 1)

    def test_fake_request_is_not_actual_batch_32(self):
        data = round_events()
        for e in data:
            if e["name"].startswith("cuda_graph.forward"):
                e["name"] = e["name"].replace("fake=0", "fake=1")
        self.assertEqual(analyze(data)["complete_rounds"], [])

    def test_strict_contract_does_not_count_padded_batch_as_32_real_requests(self):
        data = round_events()
        self.assertEqual(analyze(data, require_contract=True)["complete_rounds"], [])
        for i, width in enumerate((1, 1, 4, 4)):
            ts = 100 + i * 100
            data.append(
                event(
                    f"cuda_graph.modeling(role={i},logical_b=32,physical_b=32,q={width},"
                    f"bucket=32,capture_b=32,capture_t={32*width})",
                    ts,
                    10,
                )
            )
        self.assertEqual(
            len(analyze(data, require_contract=True)["complete_rounds"]), 1
        )
        for e in data:
            e["name"] = e["name"].replace("logical_b=32", "logical_b=31")
        self.assertEqual(analyze(data, require_contract=True)["complete_rounds"], [])


if __name__ == "__main__":
    unittest.main()
