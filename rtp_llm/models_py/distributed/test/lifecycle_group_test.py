import unittest
from unittest import mock

from rtp_llm.models_py.distributed import lifecycle_group as lg


class LifecycleGroupTest(unittest.TestCase):
    def setUp(self):
        self.state = mock.patch.object(lg, "_group", None)
        self.state.start()
        self.addCleanup(self.state.stop)

    def test_single_rank_never_creates_group(self):
        with mock.patch.object(lg.dist, "ProcessGroupGloo") as create:
            lg.init_lifecycle_group(mock.Mock(), 0, 1, 3)
        create.assert_not_called()
        self.assertIsNone(lg.get_lifecycle_group())

    def test_group_is_dedicated_and_initialized_only_once(self):
        with mock.patch.object(
            lg.dist, "is_gloo_available", return_value=True
        ), mock.patch.object(lg.dist, "PrefixStore") as prefix, mock.patch.object(
            lg.dist, "ProcessGroupGloo"
        ) as create, mock.patch.object(
            lg.dist, "new_group"
        ) as model_group, mock.patch.object(
            lg.dist, "all_reduce"
        ) as reduce:
            store = mock.Mock()
            lg.init_lifecycle_group(store, 1, 4, 9)
            self.assertIs(lg.get_lifecycle_group(), create.return_value)
            self.assertEqual(create.call_args.args[1:3], (1, 4))
            self.assertEqual(create.call_args.args[3].total_seconds(), 9)
            prefix.assert_called_once_with("rtp_llm_execution_control/", store)
            with self.assertRaisesRegex(RuntimeError, "already initialized"):
                lg.init_lifecycle_group(store, 1, 4, 9)
            create.assert_called_once()
            model_group.assert_not_called()
            reduce.assert_not_called()

    def test_unavailable_gloo_fails_before_model_start(self):
        with mock.patch.object(lg.dist, "is_gloo_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "requires Gloo"):
                lg.init_lifecycle_group(mock.Mock(), 0, 2, 3)
        self.assertIsNone(lg.get_lifecycle_group())


if __name__ == "__main__":
    unittest.main()
