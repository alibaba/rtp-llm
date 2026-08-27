import unittest
from unittest import TestCase
from unittest.mock import patch

from rtp_llm.config.engine_config import (
    GRAMMAR_MAX_COMPILE_THREADS,
    GRAMMAR_MIN_COMPILE_THREADS,
    derive_grammar_compile_threads,
)


class _FakeGrammarConfig:
    def __init__(self, num_workers: int):
        self.num_workers = num_workers


class _FakeParallelismConfig:
    def __init__(self, local_world_size: int, world_size: int):
        self.local_world_size = local_world_size
        self.world_size = world_size


class DeriveGrammarCompileThreadsTest(TestCase):
    def _derive(self, cores, local_world_size, world_size, num_workers=0):
        grammar_config = _FakeGrammarConfig(num_workers)
        with patch(
            "rtp_llm.config.engine_config.os.sched_getaffinity",
            return_value=set(range(cores)),
        ):
            derive_grammar_compile_threads(
                grammar_config,
                _FakeParallelismConfig(local_world_size, world_size),
            )
        return grammar_config.num_workers

    def test_explicit_value_is_left_alone(self):
        self.assertEqual(
            self._derive(cores=128, local_world_size=8, world_size=8, num_workers=3), 3
        )

    def test_splits_cores_across_ranks_on_the_node(self):
        self.assertEqual(self._derive(cores=128, local_world_size=8, world_size=8), 16)

    def test_one_rank_per_container_owns_its_whole_cpuset(self):
        # A multi-node job with a single rank per container: the rank is alone in its
        # cpuset, so the job's size must not shrink its budget.
        self.assertEqual(
            self._derive(cores=128, local_world_size=1, world_size=8),
            GRAMMAR_MAX_COMPILE_THREADS,
        )

    def test_node_wider_than_the_job_does_not_shrink_the_budget(self):
        # A launcher may export the node's device count even for a single-rank deploy.
        self.assertEqual(
            self._derive(cores=128, local_world_size=8, world_size=1),
            GRAMMAR_MAX_COMPILE_THREADS,
        )

    def test_single_rank_keeps_the_whole_budget(self):
        self.assertEqual(
            self._derive(cores=128, local_world_size=1, world_size=1),
            GRAMMAR_MAX_COMPILE_THREADS,
        )

    def test_negative_value_is_derived(self):
        self.assertEqual(
            self._derive(cores=128, local_world_size=8, world_size=8, num_workers=-1),
            16,
        )

    def test_clamped_below_on_a_small_cpuset(self):
        # The lower bound overrides this rank's share, so the ranks together oversubscribe
        # the cpuset. That is deliberate: a narrower fanout leaves a compile too slow to
        # finish inside the caller's budget.
        self.assertEqual(
            self._derive(cores=16, local_world_size=8, world_size=8),
            GRAMMAR_MIN_COMPILE_THREADS,
        )

    def test_never_exceeds_the_cpuset(self):
        self.assertEqual(self._derive(cores=4, local_world_size=1, world_size=1), 4)

    def test_never_exceeds_a_tiny_cpuset_shared_by_many_ranks(self):
        self.assertEqual(self._derive(cores=4, local_world_size=8, world_size=8), 4)

    def test_clamped_above_on_a_large_cpuset(self):
        self.assertEqual(
            self._derive(cores=512, local_world_size=2, world_size=2),
            GRAMMAR_MAX_COMPILE_THREADS,
        )

    def test_degenerate_rank_counts_do_not_divide_by_zero(self):
        self.assertEqual(
            self._derive(cores=64, local_world_size=0, world_size=0),
            GRAMMAR_MAX_COMPILE_THREADS,
        )


if __name__ == "__main__":
    unittest.main()
