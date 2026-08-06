"""Real-torch_memory_saver GPU integration test for the level-2 weights region.

Unlike ``weight_memory_saver_test.py`` (which fakes torch_memory_saver via
sys.modules and is GPU-free), this test exercises the *real* torch_memory_saver
on a CUDA device to prove the segment-isolation invariant that level-2 sleep
depends on:

    A persistent NON-weight tensor allocated OUTSIDE ``weights_region`` (i.e.
    from the default caching-allocator pool) is NOT discarded by
    ``pause("weights")`` / blanked by ``resume("weights")``.

This is the invariant a code reviewer flagged as unproven: level-2 discards the
whole "weights" tag without a backup, so if a live non-weight tensor ever shared
a weights-tagged physical segment it would silently become garbage after wake
(the reload only restores ``ModelWeights``). torch_memory_saver routes every
region allocation into its own ``MemPool`` while default-pool allocations stay
out of it, and :func:`loader.ModelLoader.load_weights` additionally runs a gated
``empty_cache()`` after the load so freed weight-load intermediates cannot be
reused out-of-tag. This test locks that behavior down empirically.

It uses ``hook_mode="torch"`` (CUDAPluggableAllocator) so it runs in-process
without the LD_PRELOAD shim, then drives the *real* RTP-LLM wrapper
(``weights_region`` / ``pause_weights`` / ``resume_weights``) at level 2.
Skips cleanly when CUDA or torch_memory_saver is unavailable.
"""

import os
import unittest

import torch

from rtp_llm.model_loader import weight_memory_saver as wms

_TMS_MODULE = "torch_memory_saver"


def _tms_torch_mode_available() -> bool:
    if not torch.cuda.is_available():
        return False
    # TorchMemorySaver refuses to run with expandable_segments (see _sanity_checks).
    if "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        return False
    try:
        from torch_memory_saver.entrypoint import TorchMemorySaver  # noqa: F401
    except Exception:
        return False
    return True


@unittest.skipUnless(
    _tms_torch_mode_available(),
    "requires CUDA + torch_memory_saver (torch hook mode, no expandable_segments)",
)
class Level2WeightsRegionIsolationTest(unittest.TestCase):
    """Real-TMS level-2 pause/resume must not touch out-of-region memory."""

    # A torch-hook-mode saver owns a torch.cuda.MemPool whose destructor
    # release_block()s TMS-allocated blocks; destroying more than one such pool
    # in a single process segfaults at teardown (the known
    # "MemPool destroy crashes under TMS" issue). So build exactly ONE saver for
    # the whole class, reuse it across tests, and never destroy it (the process
    # exits cleanly with a single live pool).
    _saver = None

    @classmethod
    def setUpClass(cls) -> None:
        from torch_memory_saver.entrypoint import TorchMemorySaver

        saver = TorchMemorySaver()
        saver.hook_mode = "torch"  # CUDAPluggableAllocator MemPool, no LD_PRELOAD
        cls._saver = saver

    def setUp(self) -> None:
        self._saved_env = {
            wms.ENV_SWITCH: os.environ.get(wms.ENV_SWITCH),
            wms.ENV_LEVEL: os.environ.get(wms.ENV_LEVEL),
        }
        wms._reset_for_testing()
        # Level 2: region opened with enable_cpu_backup=False (discard, no backup).
        wms.configure_from_runtime(True, 2)
        # Inject the shared torch-mode saver as the singleton wms will pick up.
        wms._tms = type(self)._saver
        wms._import_attempted = True

    def tearDown(self) -> None:
        # Never leave the region paused for the next test; do NOT empty_cache or
        # destroy the pool here (that is what triggers the teardown segfault).
        try:
            wms.resume_weights()
        except Exception:
            pass
        wms._reset_for_testing()
        for name, value in self._saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_out_of_region_tensor_survives_level2_pause_resume(self) -> None:
        self.assertEqual(wms.sleep_mode_level(), 2)
        self.assertTrue(wms.is_available())

        # A resident "weight" allocated inside the level-2 weights region, plus a
        # transient intermediate that is freed inside the region (mimics the raw
        # read / dequant / TP-split scratch that WeightModule.load produces).
        with wms.weights_region():
            weight = torch.ones(1024, 1024, device="cuda", dtype=torch.float32)
            scratch = torch.full((1024, 1024), 7.0, device="cuda")
            del scratch
        weight_ptr = weight.data_ptr()
        weight_sum_before = float(weight.sum().item())
        self.assertEqual(weight_sum_before, 1024.0 * 1024.0)

        # The gated post-load empty_cache (loader.ModelLoader.load_weights): return
        # freed weight-load intermediates so nothing reuses those blocks out-of-tag.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Persistent NON-weight sentinel, allocated OUTSIDE any region (default
        # pool). This stands in for any runtime buffer that outlives a sleep.
        sentinel = torch.full((1024, 1024), 3.5, device="cuda", dtype=torch.float32)
        sentinel_ptr = sentinel.data_ptr()
        sentinel_sum_before = float(sentinel.sum().item())

        # level-2 sleep: discard the weights tag (no cpu backup).
        self.assertTrue(wms.pause_weights())
        self.assertTrue(wms.is_paused())
        # level-2 wake: remap blank physical pages at the SAME VA. The real reload
        # would copy_ weights back from the checkpoint here; we only assert the
        # memory-plumbing invariants.
        self.assertTrue(wms.resume_weights())
        self.assertFalse(wms.is_paused())
        torch.cuda.synchronize()

        # 1) Weight VA is preserved (C++ aliases / captured CUDA graphs stay valid).
        self.assertEqual(weight.data_ptr(), weight_ptr)
        # 2) Weight content was genuinely discarded (level-2 keeps no backup): the
        #    remapped pages are blank, proving the tag really released memory.
        self.assertEqual(float(weight.sum().item()), 0.0)
        # 3) THE INVARIANT: the out-of-region sentinel is untouched by
        #    pause/resume("weights") -- same VA, same bytes.
        self.assertEqual(sentinel.data_ptr(), sentinel_ptr)
        self.assertEqual(float(sentinel.sum().item()), sentinel_sum_before)

    def test_weight_content_restored_when_refilled_after_wake(self) -> None:
        """After wake the resident tensor is writable in place at its stable VA.

        Mirrors what the level-2 reload does: copy_ fresh values into the live
        weight tensor after resume, and confirm they stick (the tensor's storage
        is validly remapped, not dangling).
        """
        with wms.weights_region():
            weight = torch.zeros(512, 512, device="cuda", dtype=torch.float32)
        weight_ptr = weight.data_ptr()

        self.assertTrue(wms.pause_weights())
        self.assertTrue(wms.resume_weights())
        torch.cuda.synchronize()

        self.assertEqual(weight.data_ptr(), weight_ptr)
        weight.copy_(torch.full_like(weight, 2.0))
        torch.cuda.synchronize()
        self.assertEqual(float(weight.sum().item()), 2.0 * 512.0 * 512.0)


if __name__ == "__main__":
    import sys

    # The torch-hook-mode saver holds a torch.cuda.MemPool that cannot be safely
    # destroyed in-process after a pause/resume cycle: its destructor segfaults in
    # release_block at interpreter finalization (the known MemPool-destroy-under-TMS
    # issue; production uses preload mode and tears the backend down by SIGKILL, so
    # this dtor never runs there). The test assertions themselves pass -- exit hard
    # with the real result before finalization so the harness sees the true outcome.
    _result = unittest.main(exit=False, verbosity=2).result
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if _result.wasSuccessful() else 1)
