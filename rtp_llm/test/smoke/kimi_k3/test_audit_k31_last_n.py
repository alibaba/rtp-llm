import copy
import unittest

from audit_k31_last_n import audit_c7_miss, audit_checkpoint_reuse, audit_prefill_states


class LastNStateAuditTest(unittest.TestCase):
    def test_c7_control_must_recompute_with_no_checkpoint_input(self):
        seed = dict(event='linear_cache_publish_candidates', request_id=1, group_tag='linear', time_ns=1,
                    checkpoints=[dict(tokens=i*1024) for i in range(8, 13)])
        start = dict(event='linear_prefill_round_completed', request_id=2, group_tag='linear',
                     time_ns=3, prefix_tokens=0, input_block=0)
        self.assertEqual(audit_c7_miss({0: [seed, start]}, [1], [2])[0]['reused_tokens'], 0)
        for changed in (dict(start, prefix_tokens=7168, input_block=7), dict(start, input_block=8)):
            with self.assertRaisesRegex(ValueError, 'recompute'):
                audit_c7_miss({0: [seed, changed]}, [1], [2])
        match = dict(event='linear_cache_matched', request_id=2, group_tag='linear',
                     time_ns=2, checkpoint_tokens=7168)
        with self.assertRaisesRegex(ValueError, 'recompute'):
            audit_c7_miss({0: [seed, match, start]}, [1], [2])

    def test_tier_match_requires_actual_consumed_state_and_completed_lower_tier_load(self):
        for tier in ("DEVICE", "HOST", "DISK"):
            seed = dict(event="linear_cache_publish_candidates", request_id=1, group_tag="linear", time_ns=1,
                        checkpoints=[dict(tokens=i*1024, block=i, cache_key=i) for i in range(8, 13)])
            match = dict(event="linear_cache_matched", request_id=2, group_tag="linear", time_ns=2,
                         checkpoint_tokens=10240, source_tier=tier, source_block=10, cache_key=10)
            loaded = dict(event="linear_cache_loaded", request_id=2, group_tag="linear", time_ns=3,
                          checkpoint_tokens=10240, source_tier=tier, device_block=10)
            first = dict(event="linear_prefill_round_completed", request_id=2, group_tag="linear", time_ns=4,
                         prefix_tokens=10240, input_block=10)
            written = dict(event='linear_cache_transfer_completed', group_tag='linear', time_ns=2,
                           cache_key=10, target_tier=tier, target_block=10, payload_bytes=4096,
                           crc_enabled=True)
            rows = [seed, *([written] if tier != 'DEVICE' else []), match,
                    *([loaded] if tier != "DEVICE" else []), first]
            with self.subTest(tier=tier):
                self.assertEqual(audit_checkpoint_reuse({0: rows}, [1], [2], tier)[0]["source_tier"], tier)
                for bad in (dict(first, input_block=11), dict(first, prefix_tokens=9216)):
                    with self.assertRaises(ValueError):
                        audit_checkpoint_reuse({0: rows[:-1]+[bad]}, [1], [2], tier)
                if tier != "DEVICE":
                    with self.assertRaisesRegex(ValueError, 'completed write'):
                        audit_checkpoint_reuse({0: [seed, dict(written, crc_enabled=False),
                            match, loaded, first]}, [1], [2], tier)
                    with self.assertRaisesRegex(ValueError, "completed load"):
                        audit_checkpoint_reuse({0: [seed, match, first]}, [1], [2], tier)
                    with self.assertRaisesRegex(ValueError, 'completed write'):
                        audit_checkpoint_reuse({0: [seed, match, loaded, first]}, [1], [2], tier)

    def fixture(self, n=5, tail=0):
        prompt, interval, chunk = 1_048_576 + tail, 1024, 65_536
        selected = {(1024-n+index+1)*interval: index+1 for index in range(n)}
        borrowed = n+1 if tail else n
        allocation = dict(event="linear_prefill_allocated", request_id=17, group_tag="linear",
                          time_ns=1, prompt_tokens=prompt, initial_reuse_tokens=0,
                          configured_tail_blocks=n, page_tokens=128, checkpoint_tokens=interval,
                          planned_new_blocks=n+bool(tail), owned_state_blocks=n+bool(tail),
                          has_separate_tail=bool(tail), borrowed_block=borrowed)
        events = {rank: [] for rank in range(8)}
        events[0].append(allocation)
        prefix, previous, tick = 0, 0, 10
        while prefix < prompt:
            end = min(prompt, prefix+chunk)
            writes = [dict(block=block, tokens=point, retained=True)
                      for point, block in selected.items() if prefix < point <= end]
            if end not in selected:
                writes.append(dict(block=borrowed, tokens=end, retained=False))
            if previous and previous in [write["block"] for write in writes]:
                for rank in events:
                    events[rank].append(dict(event="linear_prefill_conv_snapshot_submitted",
                        world_rank=rank, time_ns=tick, rows=[dict(prefix_tokens=prefix,
                        query_tokens=end-prefix, input_block=previous)]))
            events[0].append(dict(event="linear_prefill_round_completed", request_id=17,
                group_tag="linear", time_ns=tick+1, prefix_tokens=prefix, end_tokens=end,
                input_block=previous, output_block=writes[-1]["block"], writes=writes))
            prefix, previous, tick = end, writes[-1]["block"], tick+10
        events[0].append(dict(event="linear_cache_publish_candidates", request_id=17,
            group_tag="linear", time_ns=tick, target_tier="HOST", prompt_tokens=prompt,
            checkpoints=[dict(tokens=point, block=block) for point, block in selected.items()]))
        return events

    def test_last_one_through_five_with_and_without_tail(self):
        for n in range(1, 6):
            for tail in (0, 128):
                with self.subTest(n=n, tail=tail):
                    proof = audit_prefill_states(self.fixture(n, tail), tail_blocks=n,
                                                require_one_million=not tail)
                    self.assertEqual(proof["requests"][0]["new_state_blocks"], n+bool(tail))

    def test_rejects_work_bank_concurrent_writers_and_wrong_input(self):
        for corruption in ("allocation", "writers", "input"):
            events = self.fixture()
            rounds = [e for e in events[0] if e["event"] == "linear_prefill_round_completed"]
            if corruption == "allocation":
                events[0][0]["owned_state_blocks"] = 70
            elif corruption == "writers":
                rounds[-1]["writes"][0]["block"] = rounds[-1]["writes"][-1]["block"]
            else:
                rounds[1]["input_block"] = 2
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                audit_prefill_states(events)

    def test_requires_snapshot_on_each_rank(self):
        events = self.fixture()
        events[7] = []
        with self.assertRaisesRegex(ValueError, "rank7"):
            audit_prefill_states(events)

    def test_rejects_early_or_wrong_key_publication(self):
        for change in ({"time_ns": 5}, {"checkpoints": [dict(tokens=65536, block=5)]}):
            events = self.fixture()
            events[0][-1].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "uninitialized"):
                audit_prefill_states(events)

    def test_tail_and_wrong_rank_cannot_count_as_publication_or_snapshot(self):
        events = self.fixture(tail=128)
        events[0][-1]["checkpoints"][-1] = dict(tokens=1_048_704, block=6)
        with self.assertRaisesRegex(ValueError, "uninitialized"):
            audit_prefill_states(events, require_one_million=False)
        events = self.fixture()
        events[7] = copy.deepcopy(events[6])
        with self.assertRaisesRegex(ValueError, "rank7"):
            audit_prefill_states(events)


if __name__ == "__main__":
    unittest.main()
