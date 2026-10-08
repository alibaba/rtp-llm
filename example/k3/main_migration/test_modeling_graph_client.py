import copy
import threading
import unittest

from run_modeling_graph_case import arm_profile, full_batch_ready, owner_rows


def owner(rank, tp=4):
    return {
        "alive": True,
        "dp_rank": rank,
        "dp_size": 8 // tp,
        "tp_size": tp,
        "running_task_info": [
            {"input_length": 65536, "iterate_count": 10, "is_waiting": False}
            for _ in range(32)
        ],
    }


class ServingCaptureBoundaryTest(unittest.TestCase):
    def test_profiles_every_tp_owner_not_just_owner_zero(self):
        calls = []
        lock = threading.Lock()

        def request(url, payload):
            with lock:
                calls.append((url, payload))
            return {"status": "ok"}

        urls = ["http://decode:31100", "http://decode:31136"]
        result = arm_profile(urls, "test_only", 40, request)
        self.assertEqual(len(result), 2)
        self.assertEqual(
            {url for url, _ in calls}, {url + "/start_profile" for url in urls}
        )
        self.assertTrue(all(payload["enable_all_rank"] for _, payload in calls))

    def test_missing_owner_or_partial_batch_cannot_start_measurement(self):
        with self.assertRaisesRegex(RuntimeError, "every Decode owner"):
            owner_rows([owner(0)], 4)
        rows = owner_rows([{"results": [owner(0), owner(1)]}], 4)
        self.assertTrue(full_batch_ready(rows))
        for mutation in ("partial", "waiting", "wrong_kv"):
            changed = copy.deepcopy(rows)
            if mutation == "partial":
                changed[1]["running_task_info"].pop()
            else:
                field, value = {
                    "waiting": ("is_waiting", True),
                    "wrong_kv": ("input_length", 69632),
                }[mutation]
                changed[1]["running_task_info"][0][field] = value
            self.assertFalse(full_batch_ready(changed))

    def test_fixed_feat_legacy_status_has_no_root_owner_or_live_iteration_counter(self):
        statuses = [owner(0), owner(1)]
        for index, row in enumerate(statuses):
            del row["dp_rank"]
            for task in row["running_task_info"]:
                task.update(dp_rank=index, iterate_count=0)
        rows = owner_rows(statuses, 4)
        self.assertTrue(full_batch_ready(rows))
        for row in statuses:
            row["running_task_info"] = []
        self.assertEqual(set(owner_rows(statuses, 4)), {0, 1})

    def test_duplicate_frontend_snapshots_and_topology_mismatch(self):
        rows = owner_rows(
            [{"results": [owner(0), owner(1)]}, {"results": [owner(0), owner(1)]}], 4
        )
        self.assertEqual(set(rows), {0, 1})
        bad = owner(1)
        bad["tp_size"] = 8
        with self.assertRaisesRegex(RuntimeError, "topology"):
            owner_rows([owner(0), bad], 4)

    def test_failed_profile_ack_is_not_capture_success(self):
        with self.assertRaisesRegex(RuntimeError, "acknowledged"):
            arm_profile(
                ["http://decode:31100", "http://decode:31136"],
                "test_only",
                40,
                lambda url, payload: {"status": "ok" if "31100" in url else "ignored"},
            )


if __name__ == "__main__":
    unittest.main()
