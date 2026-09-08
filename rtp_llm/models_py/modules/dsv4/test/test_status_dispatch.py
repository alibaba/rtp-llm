import unittest

from rtp_llm.models_py.modules.dsv4.block import _supports_numerical_status


class _PlainCallable:
    def __call__(self, value):
        return value


class _StatusCallable:
    supports_numerical_status = True

    def __call__(self, value, *, numerical_status=None):
        return value, numerical_status


class NumericalStatusDispatchTest(unittest.TestCase):
    def test_capability_is_explicit_opt_in(self) -> None:
        self.assertFalse(_supports_numerical_status(_PlainCallable()))
        self.assertTrue(_supports_numerical_status(_StatusCallable()))

    def test_bound_method_inherits_owner_opt_in(self) -> None:
        self.assertTrue(_supports_numerical_status(_StatusCallable().__call__))


if __name__ == "__main__":
    unittest.main()
