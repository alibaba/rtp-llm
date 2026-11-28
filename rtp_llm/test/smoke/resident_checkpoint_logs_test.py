import unittest

from rtp_llm.test.smoke.resident_checkpoint_logs import cold_checkpoint_owners


class ResidentCheckpointLogsTest(unittest.TestCase):
    def test_only_vision_may_read_on_warm_start(self):
        for event in (
            "Streamed checkpoint tensors with fastsafetensors in 0.10s",
            "NewLoader imported rank-local FT checkpoint with scratch",
            "NewLoader imported Meta consolidated checkpoint with scratch",
        ):
            with self.subTest(event=event):
                vision = f"[INFO] [checkpoint_owner=deepseek_v41_vision] {event}"
                self.assertEqual(cold_checkpoint_owners(vision), {"deepseek_v41_vision"})
                for owner in ("deepseek_v41", "deepseek_v41_dspark"):
                    mixed = f"{vision}\n[INFO] [checkpoint_owner={owner}] {event}"
                    self.assertEqual(cold_checkpoint_owners(mixed), {"deepseek_v41_vision", owner})
                self.assertIn("<untagged>", cold_checkpoint_owners(vision + "\n" + event))

    def test_attach_is_not_a_checkpoint_import(self):
        self.assertFalse(cold_checkpoint_owners("attached resident weights gpu0/draft: 1024 bytes"))


if __name__ == "__main__":
    unittest.main()
