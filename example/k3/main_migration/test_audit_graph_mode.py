"""Reject missing or inconsistent eager Native MTP evidence."""

import unittest

from example.k3.main_migration.audit_orthogonal_smoke import audit, read_paged_contract


def fixture():
    cases, stages = [], []
    decode = {rank: [] for rank in range(8)}
    for index, size in enumerate((1, 4, 8, 63, 64)):
        start = 1000 * (index + 1)
        names = [f"batch{size}_{i}" for i in range(size)]
        cases.extend(
            {"name": name, "phase": "prepare", "decode_owner_rank": i % 2}
            for i, name in enumerate(names)
        )
        stages.append(
            {
                "name": f"orthogonal_decode_{index:02d}_batch_{size}",
                "case_names": names,
                "start_time_ns": start,
                "end_time_ns": start + 90,
            }
        )
        for rank in range(8):
            local = sum(i % 2 == rank // 4 for i in range(size))
            if not local:
                continue
            decode[rank].extend(
                [
                    {
                        "event": "mtp_target_verify_forward",
                        "input_rows": local,
                        "stream_count": local,
                        "token_rows": local * 4,
                        "time_ns": start + 10,
                    },
                    {
                        "event": "mtp_draft_decode_forward",
                        "input_rows": local,
                        "stream_count": local,
                        "propose_step": 3,
                        "time_ns": start + 20,
                    },
                ]
            )
            for phase, width in (
                ("target_verify", 4),
                ("proposal_or_decode", 1),
                ("mtp_update", 4),
            ):
                physical = ((local + 3) // 4 * 4) if width == 1 else local
                decode[rank].append(
                    {
                        "kind": "mla_paged_input_contract",
                        "time_ns": start + 5,
                        "phase": phase,
                        "graph": 0,
                        "logical_batch": local,
                        "physical_batch": physical,
                        "q": width,
                        "physical_tokens": physical * width,
                        "operand_dtype": (
                            "torch.float8_e4m3fn"
                            if phase == "target_verify"
                            else "torch.bfloat16"
                        ),
                        "projection_dtype": "bf16",
                        "backend": "tokenspeed_page_rr",
                    }
                )
    result = {
        "passed": True,
        "suite": "orthogonal-flow",
        "case_profile": "compact-64k-v1",
        "cases": cases,
        "stages": stages,
        "orthogonal_phases": ["decode"],
    }
    return result, {rank: [] for rank in range(8)}, decode


class AuditGraphModeTest(unittest.TestCase):
    def test_graph_off_requires_all_rank_fixed_width_paged_modeling(self):
        result, prefill, decode = fixture()
        verdict = audit(result, prefill, decode, decode_dp=2, decode_graph=False)
        self.assertTrue(verdict["passed"], verdict["errors"])
        self.assertFalse(verdict["decode_graph_expected"])
        self.assertFalse(audit(result, prefill, decode, decode_dp=2)["passed"])

    def test_graph_off_rejects_replay_and_missing_q4_update(self):
        result, prefill, decode = fixture()
        decode[0].append({"event": "cuda_graph_replay", "role": 4, "time_ns": 1025})
        self.assertFalse(
            audit(result, prefill, decode, decode_dp=2, decode_graph=False)["passed"]
        )
        decode[0].pop()
        decode[3] = [event for event in decode[3] if event.get("phase") != "mtp_update"]
        self.assertFalse(
            audit(result, prefill, decode, decode_dp=2, decode_graph=False)["passed"]
        )

    def test_graph_off_rejects_bad_dtype_backend_padding_and_future_contract(self):
        for key, value in (
            ("operand_dtype", "torch.float8_e4m3fn"),
            ("backend", "unpaged"),
            ("physical_batch", 33),
            ("q", 3),
            ("time_ns", 999999),
        ):
            with self.subTest(key=key):
                result, prefill, decode = fixture()
                for event in decode[7]:
                    if event.get("phase") == "mtp_update":
                        event[key] = value
                self.assertFalse(
                    audit(result, prefill, decode, decode_dp=2, decode_graph=False)[
                        "passed"
                    ]
                )

    def test_graph_off_rejects_compacted_verify_and_history_chunk(self):
        result, prefill, decode = fixture()
        for event in decode[0]:
            if event.get("event") == "mtp_target_verify_forward":
                event["token_rows"] -= 1
        self.assertFalse(
            audit(result, prefill, decode, decode_dp=2, decode_graph=False)["passed"]
        )
        result, prefill, decode = fixture()
        decode[0].append({"kind": "mla_prefix_executed", "time_ns": 1030})
        self.assertFalse(
            audit(result, prefill, decode, decode_dp=2, decode_graph=False)["passed"]
        )

    def test_parser_retains_dtype_shape_and_cst_timestamp(self):
        line = (
            "[root][2026-10-10 03:42:48.894][12][Dummy-6][INFO] "
            "[K3_MLA_PAGED_INPUT] phase=mtp_update graph=0 logical_batch=31 physical_batch=31 "
            "q=4 physical_tokens=124 operand_dtype=torch.bfloat16 projection_dtype=bf16 "
            "backend=tokenspeed_page_rr workspace=0xabc\n"
        )
        event = read_paged_contract(line)
        self.assertEqual(event["time_ns"] // 1000000, 1791574968894)
        self.assertEqual(event["physical_tokens"], 124)
        self.assertEqual(event["operand_dtype"], "torch.bfloat16")
        with self.assertRaises(ValueError):
            read_paged_contract(line.replace("[2026-10-10 03:42:48.894]", ""))


if __name__ == "__main__":
    unittest.main()
