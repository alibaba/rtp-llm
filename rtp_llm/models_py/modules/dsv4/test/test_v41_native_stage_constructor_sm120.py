"""Native maintained 21/19 constructors with owned-only weights and CP4 frames."""

import socket
import unittest
from dataclasses import replace

import torch
import torch.multiprocessing as mp

from rtp_llm.models_py.modules.dsv4.test.test_native_l20_transfer import TEXT
from rtp_llm.models_py.modules.dsv4.test.test_v41_cp_stage_slices_sm120 import run_rank
from rtp_llm.models_py.modules.dsv4.transformer import V4Args, V4Transformer


class UnreadableWeights:
    @property
    def global_weights(self):
        raise AssertionError("unsupported stage consumed global weights")

    @property
    def weights(self):
        raise AssertionError("unsupported stage consumed layer weights")


def args_for(rank):
    return V4Args(
        dim=5120,
        n_layers=40,
        n_mtp_layers=0,
        n_hash_layers=0,
        pp_size=2,
        pp_rank=rank,
        pp_layer_ids=list(range(21)) if rank == 0 else list(range(21, 40)),
        pp_has_embedding=rank == 0,
        pp_has_lm_head=rank == 1,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20,
        v41_config=TEXT,
        gen_num_per_cycle=0,
    )


class NativeStageAdmissionTest(unittest.TestCase):
    def test_wrong_rosters_and_roles_fail_before_weights(self):
        for rank in (0, 1):
            args = args_for(rank)
            for bad in (
                replace(args, pp_size=3),
                replace(args, pp_rank=2),
                replace(args, n_layers=6),
                replace(args, pp_layer_ids=None),
                replace(args, pp_layer_ids=list(reversed(args.pp_layer_ids))),
                replace(args, pp_layer_ids=list(range(20 * rank, 20 * rank + 20))),
                replace(args, pp_layer_ids=[float(i) for i in args.pp_layer_ids]),
                replace(args, pp_has_embedding=not args.pp_has_embedding),
                replace(args, pp_has_lm_head=not args.pp_has_lm_head),
                replace(args, is_decode_role=True),
                replace(args, gen_num_per_cycle=1),
                replace(args, n_mtp_layers=1),
                replace(args, commit_only=True),
                replace(args, fp8_kv_cache=False),
                replace(args, tp_size=4),
                replace(args, ep_size=8),
                replace(args, dim=4096),
                replace(args, compress_ratios=[0] * 40),
            ):
                with self.subTest(rank=rank, bad=bad), self.assertRaises(
                    NotImplementedError
                ):
                    V4Transformer(bad, UnreadableWeights())

    def test_wrong_source_and_engram_ownership_fail_before_weights(self):
        args = args_for(1)
        for config in (
            {**TEXT, "candidate_source_layer_id": 24},
            {**TEXT, "kv_source_layer_ids": [2, 8, 14, 24]},
            {**TEXT, "index_source_layer_ids": [2, 8, 14, 20, 25, 28, 32, 36]},
            {**TEXT, "engram_layer_ids": [1, 21]},
        ):
            with self.subTest(config=config), self.assertRaises(
                (ValueError, NotImplementedError)
            ):
                V4Transformer(replace(args, v41_config=config), UnreadableWeights())


class NativeStageConstructorSM120Test(unittest.TestCase):
    def test_native_21_19_construction_with_imported_cp_frames(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 8)
        for rank in range(8):
            self.assertEqual(torch.cuda.get_device_capability(rank), (12, 0))
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        mp.spawn(run_rank, args=(port, True), nprocs=8, join=True)


if __name__ == "__main__":
    unittest.main()
