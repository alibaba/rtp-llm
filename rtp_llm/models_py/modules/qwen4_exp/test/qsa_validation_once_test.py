"""CPU contracts for one-forward side-table validation reuse.

These execute the real CPU dual-table verdict and failure path; they do not
prove CUDA stream ordering or device assertions. The GPU geometry tests in
qsa_dual_table_verdict_test cover the actual two-kernel submission path.
"""

import threading
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.model_desc.qwen4_exp_runtime import qsa_step_plan as plans


class QSAValidationOnceTest(unittest.TestCase):
    def setUp(self):
        strict = patch.dict("os.environ", {"RTP_LLM_QSA_STRICT_VALIDATION": "0"})
        strict.start()
        self.addCleanup(strict.stop)
        self.kv = torch.tensor([[1, 2, 0]], dtype=torch.int32)
        self.state = torch.tensor([[2, 3, 0]], dtype=torch.int32)
        self.step = SimpleNamespace(
            mode=plans.MODE_PREFIX_REUSE,
            main_inputs=SimpleNamespace(is_cuda_graph=False),
            required_kv_columns=torch.tensor([2], dtype=torch.int32),
            required_state_columns_device=torch.tensor([2], dtype=torch.int32),
        )
        verdict_patch = patch.object(
            plans, "_side_tables_verdict", wraps=plans._side_tables_verdict
        )
        self.verdict = verdict_patch.start()
        self.addCleanup(verdict_patch.stop)

    def validate(self, *, kv_capacity=8, state_capacity=8):
        plans.validate_side_tables_for_step(
            self.step,
            self.kv,
            self.state,
            kv_pool_blocks=kv_capacity,
            state_pool_blocks=state_capacity,
        )

    def test_first_call_checks_later_layers_reuse_and_next_forward_rechecks(self):
        for expected in (1, 2):
            with plans.qsa_validation_step():
                for _ in range(12):
                    self.validate()
                self.assertEqual(self.verdict.call_count, expected)
        # No explicit model-forward epoch means no reuse, even for this plan.
        self.validate()
        self.validate()
        self.assertEqual(self.verdict.call_count, 4)

    def test_each_table_rebind_and_normal_version_change_rechecks(self):
        with plans.qsa_validation_step():
            self.validate()
            self.kv = self.kv.clone()
            self.validate()
            self.state = self.state.clone()
            self.validate()
            self.kv[0, 0] = 2
            self.validate()
            self.state[0, 0] = 1
            self.validate()
            self.assertEqual(self.verdict.call_count, 5)

    def test_required_vectors_rebind_and_mutation_recheck(self):
        with plans.qsa_validation_step():
            self.validate()
            self.step.required_kv_columns = self.step.required_kv_columns.clone()
            self.validate()
            self.step.required_state_columns_device = (
                self.step.required_state_columns_device.clone()
            )
            self.validate()
            self.step.required_kv_columns.fill_(1)
            self.validate()
            self.step.required_state_columns_device.fill_(1)
            self.validate()
            self.assertEqual(self.verdict.call_count, 5)

    def test_each_capacity_and_plan_owner_is_part_of_the_key(self):
        with plans.qsa_validation_step():
            self.validate()
            self.validate(kv_capacity=9)
            self.validate(state_capacity=9)
            self.step = SimpleNamespace(**vars(self.step))
            self.validate()
            self.assertEqual(self.verdict.call_count, 4)
            # A smaller pool must still fail before its writer can run.
            with self.assertRaisesRegex(RuntimeError, "indexer_state"):
                self.validate(state_capacity=3)

    def test_alias_layout_and_data_rebinding_do_not_reuse(self):
        with plans.qsa_validation_step():
            self.validate()
            # Same owner and address, changed stride (Tensor.data need not bump
            # _version). The table has the same shape and valid integer values.
            self.kv.data = self.kv.as_strided((1, 3), (1, 1))
            self.validate()
            self.kv.data = self.kv.clone()
            self.validate()
            self.assertEqual(self.verdict.call_count, 3)

    def test_changed_invalid_table_raises_and_is_not_cached(self):
        with plans.qsa_validation_step():
            self.validate()
            self.kv[0, 0] = 0
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "indexer_kv"):
                    self.validate()
            self.assertEqual(self.verdict.call_count, 3)

    def test_assert_dispatch_failure_does_not_publish_a_cache_entry(self):
        with plans.qsa_validation_step():
            with patch.object(
                plans, "qsa_assert_valid", side_effect=RuntimeError("dispatch")
            ):
                with self.assertRaisesRegex(RuntimeError, "dispatch"):
                    self.validate()
            self.validate()
            self.validate()
            self.assertEqual(self.verdict.call_count, 2)

    def test_exception_exit_and_nested_scopes_restore_epoch(self):
        with plans.qsa_validation_step():
            self.validate()
            with self.assertRaisesRegex(RuntimeError, "nested"):
                with plans.qsa_validation_step():
                    self.validate()
                    raise RuntimeError("nested")
            self.validate()
            self.assertEqual(self.verdict.call_count, 2)
        with self.assertRaisesRegex(RuntimeError, "outer"):
            with plans.qsa_validation_step():
                self.validate()
                raise RuntimeError("outer")
        self.validate()
        self.assertEqual(self.verdict.call_count, 4)

    def test_inference_metadata_revalidates_in_new_epoch(self):
        with torch.inference_mode():
            self.kv = self.kv.clone()
            self.state = self.state.clone()
            self.step.required_kv_columns = self.step.required_kv_columns.clone()
            self.step.required_state_columns_device = (
                self.step.required_state_columns_device.clone()
            )
            with plans.qsa_validation_step():
                self.validate()
                self.validate()
            # The same inference tensor object/address can be reused next
            # forward; mutation BETWEEN epochs must not inherit its old check.
            self.kv[0, 0] = 0
            with plans.qsa_validation_step():
                with self.assertRaisesRegex(RuntimeError, "indexer_kv"):
                    self.validate()
            self.assertEqual(self.verdict.call_count, 2)

    def test_scope_holds_owners_until_exit(self):
        with plans.qsa_validation_step():
            self.validate()
            self.verdict.reset_mock()  # Do not let mock.call_args retain owners.
            old_kv = weakref.ref(self.kv)
            self.kv = self.kv.clone()
            self.assertIsNotNone(old_kv())
        self.assertIsNone(old_kv())

    def test_stream_changes_need_an_independent_submission(self):
        with plans.qsa_validation_step():
            with patch.object(
                plans, "_side_validation_stream_key", return_value=("cuda", 7)
            ):
                self.validate()
                self.validate()
            with patch.object(
                plans, "_side_validation_stream_key", return_value=("cuda", 9)
            ):
                self.validate()
                self.validate()
            self.assertEqual(self.verdict.call_count, 2)

    def test_threads_do_not_inherit_another_threads_epoch(self):
        errors = []

        def worker():
            try:
                # No scope on this thread: both calls must submit.
                self.validate()
                self.validate()
            except BaseException as error:
                errors.append(error)

        with plans.qsa_validation_step():
            self.validate()
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
            self.validate()
        self.assertEqual(errors, [])
        self.assertEqual(self.verdict.call_count, 3)

    def test_strict_graph_and_draft_modes_do_not_reuse(self):
        with plans.qsa_validation_step():
            with patch.dict("os.environ", {"RTP_LLM_QSA_STRICT_VALIDATION": "1"}):
                self.validate()
                self.validate()
            self.step.main_inputs.is_cuda_graph = True
            self.validate()
            self.validate()
            self.step.main_inputs.is_cuda_graph = False
            self.step.mode = plans.MODE_DRAFT_PREFIX_REUSE
            self.validate()
            self.validate()
            self.assertEqual(self.verdict.call_count, 6)

    def test_capture_entry_does_not_create_a_reusable_epoch(self):
        with patch("torch.cuda.is_initialized", return_value=True), patch(
            "torch.cuda.is_current_stream_capturing", return_value=True
        ):
            with plans.qsa_validation_step():
                self.validate()
                self.validate()
        self.assertEqual(self.verdict.call_count, 2)


if __name__ == "__main__":
    unittest.main()
