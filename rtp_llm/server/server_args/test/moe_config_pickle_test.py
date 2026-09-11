import ast
import copyreg
import pickle
import unittest
from pathlib import Path

from rtp_llm.config.moe_config import B12X_ZEROED_ENERGY_LIMIT_DEFAULT
from rtp_llm.ops import MoeConfig


class _InvalidMoeConfigState(MoeConfig):
    def __reduce__(self):
        return copyreg.__newobj__, (_InvalidMoeConfigState,), (False,)


_LEGACY_MOE_CONFIG_STATE = (
    True,
    False,
    False,
    True,
    True,
    False,
    True,
    17,
    512,
    True,
    23,
    "fp4_no_dp",
)


class _LegacyMoeConfigState(MoeConfig):
    def __reduce__(self):
        return (
            copyreg.__newobj__,
            (_LegacyMoeConfigState,),
            _LEGACY_MOE_CONFIG_STATE,
        )


_PREVIOUS_MOE_CONFIG_STATE = _LEGACY_MOE_CONFIG_STATE + (
    "b12x",
    0.25,
    True,
)


class _PreviousMoeConfigState(MoeConfig):
    def __reduce__(self):
        return (
            copyreg.__newobj__,
            (_PreviousMoeConfigState,),
            _PREVIOUS_MOE_CONFIG_STATE,
        )


def _bound_fields():
    return sorted(
        name for name, value in vars(MoeConfig).items() if isinstance(value, property)
    )


def _stub_fields():
    lines = Path("rtp_llm/ops/libth_transformer_config.pyi").read_text().splitlines()
    start = next(
        index for index, line in enumerate(lines) if line == "class MoeConfig:"
    )
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if lines[index] and not lines[index].startswith((" ", "\t"))
        ),
        len(lines),
    )
    stub = ast.parse("\n".join(lines[start:end]), filename="MoeConfig.pyi")
    moe_config = next(
        node
        for node in stub.body
        if isinstance(node, ast.ClassDef) and node.name == "MoeConfig"
    )
    return sorted(
        statement.target.id
        for statement in moe_config.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
    )


class MoeConfigPickleTest(unittest.TestCase):
    def test_type_stub_covers_all_bound_fields(self):
        self.assertEqual(_stub_fields(), _bound_fields())

    def test_python_and_cpp_defaults_match(self):
        self.assertEqual(
            MoeConfig().b12x_zeroed_energy_limit,
            B12X_ZEROED_ENERGY_LIMIT_DEFAULT,
        )

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
                    elif isinstance(current, float):
                        value = 0.125 + index
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

    def test_legacy_state_restores_old_fields_and_defaults_new_fields(self):
        defaults = MoeConfig()
        restored = pickle.loads(
            pickle.dumps(_LegacyMoeConfigState(), protocol=pickle.HIGHEST_PROTOCOL)
        )

        legacy_fields = (
            "use_deepep_moe",
            "use_deepep_internode",
            "use_deepep_low_latency",
            "use_deepep_p2p_low_latency",
            "use_mori_ep",
            "fake_balance_expert",
            "hack_moe_expert",
            "deep_ep_num_sm",
            "masked_max_token_num",
            "use_all_gather",
            "ll_num_max_token",
            "moe_strategy",
        )
        self.assertEqual(
            tuple(getattr(restored, field) for field in legacy_fields),
            _LEGACY_MOE_CONFIG_STATE,
        )
        self.assertEqual(restored.fp4_moe_op, defaults.fp4_moe_op)
        self.assertEqual(
            restored.b12x_zeroed_energy_limit,
            defaults.b12x_zeroed_energy_limit,
        )

    def test_previous_15_field_state_ignores_removed_field(self):
        restored = pickle.loads(
            pickle.dumps(_PreviousMoeConfigState(), protocol=pickle.HIGHEST_PROTOCOL)
        )

        self.assertEqual(restored.fp4_moe_op, "b12x")
        self.assertEqual(restored.b12x_zeroed_energy_limit, 0.25)

    def test_invalid_state_reports_expected_and_actual_field_count(self):
        expected_count = len(_bound_fields())
        serialized = pickle.dumps(_InvalidMoeConfigState())
        with self.assertRaisesRegex(
            RuntimeError,
            f"MoeConfig unpickle error: expected 12 or {expected_count} or 15 "
            "fields, got 1",
        ):
            pickle.loads(serialized)


if __name__ == "__main__":
    unittest.main()
