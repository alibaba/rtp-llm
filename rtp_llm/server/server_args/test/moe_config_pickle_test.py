import copyreg
import pickle
import unittest

from rtp_llm.ops import MoeConfig


class _InvalidMoeConfigState(MoeConfig):
    def __reduce__(self):
        return copyreg.__newobj__, (_InvalidMoeConfigState,), (False,)


def _bound_fields():
    return sorted(
        name for name, value in vars(MoeConfig).items() if isinstance(value, property)
    )


class MoeConfigPickleTest(unittest.TestCase):
    def test_roundtrip_preserves_all_bound_fields(self):
        fields = _bound_fields()
        self.assertTrue(fields, "MoeConfig exposes no bound fields")
        defaults = MoeConfig()
        bool_fields = [
            field for field in fields if isinstance(getattr(defaults, field), bool)
        ]
        self.assertTrue(bool_fields, "MoeConfig exposes no bool fields")

        # A one-hot pattern gives every bool field a unique signature across
        # round trips, so swapping two bool positions cannot pass unnoticed.
        for active_bool in bool_fields:
            with self.subTest(active_bool=active_bool):
                config = MoeConfig()
                expected = {}
                for index, field in enumerate(fields):
                    current = getattr(config, field)
                    if isinstance(current, bool):
                        value = field == active_bool
                    elif isinstance(current, int):
                        value = 1000 + index
                    elif isinstance(current, str):
                        value = f"pickle-test-{field}"
                    else:
                        self.fail(
                            f"unsupported MoeConfig field type: "
                            f"{field}={type(current)}"
                        )
                    setattr(config, field, value)
                    expected[field] = value

                restored = pickle.loads(
                    pickle.dumps(config, protocol=pickle.HIGHEST_PROTOCOL)
                )
                self.assertEqual(
                    {field: getattr(restored, field) for field in fields}, expected
                )

    def test_invalid_state_reports_expected_and_actual_field_count(self):
        expected_count = len(_bound_fields())
        serialized = pickle.dumps(_InvalidMoeConfigState())
        with self.assertRaisesRegex(
            RuntimeError,
            f"MoeConfig unpickle error: expected {expected_count} fields, got 1",
        ):
            pickle.loads(serialized)


if __name__ == "__main__":
    unittest.main()
