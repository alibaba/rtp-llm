import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.models.minimax_m31 import MiniMaxM31
from rtp_llm.models.minimax_m31_vl import MiniMaxM31VL
from rtp_llm.ops import RoleType


class MiniMaxM31StartupTest(unittest.TestCase):
    def test_cp_role_contract_before_model_construction(self):
        for model_cls, role, tp_size, enabled, sharded, auxiliary in product(
            (MiniMaxM31, MiniMaxM31VL),
            (RoleType.PREFILL, RoleType.DECODE, RoleType.PDFUSION),
            (1, 2, 4),
            (False, True),
            (False, True),
            (None, [0]),
        ):
            with self.subTest(
                model=model_cls.__name__,
                role=role,
                tp=tp_size,
                enabled=enabled,
                sharded=sharded,
                auxiliary=auxiliary,
            ):
                model = SimpleNamespace(
                    model_config=SimpleNamespace(
                        capture_aux_hidden_layer_ids=auxiliary,
                        eplb_config=SimpleNamespace(enable_eplb=lambda: False),
                        moe_prefill_max_tokens_per_rank=None,
                        max_seq_len=32768,
                    ),
                    parallelism_config=SimpleNamespace(
                        role_type=role,
                        dp_size=1,
                        tp_size=tp_size,
                        prefill_cp_config=SimpleNamespace(
                            is_enabled=lambda: enabled,
                            kv_cache_sharded=sharded,
                        ),
                    ),
                    weight=object(),
                    moe_config=object(),
                    max_generate_batch_size=32,
                    fmha_config=None,
                    hw_kernel_config=None,
                )
                # Exercise real startup validation without importing CUDA kernels.
                constructor = Mock()
                with (
                    patch("torch.cuda.is_available", return_value=True),
                    patch("torch.cuda.get_device_capability", return_value=(10, 3)),
                    patch.dict(
                        "sys.modules",
                        {
                            "rtp_llm.models_py.model_desc.minimax_m31_model": SimpleNamespace(
                                MiniMaxM31Model=constructor
                            ),
                        },
                    ),
                ):
                    if enabled and role != RoleType.DECODE and tp_size == 1:
                        with self.assertRaisesRegex(ValueError, "requires TP>1"):
                            model_cls._create_python_model(model)
                        constructor.assert_not_called()
                    elif enabled and sharded and role == RoleType.PDFUSION:
                        with self.assertRaisesRegex(ValueError, "sharded CP.*PDFUSION"):
                            model_cls._create_python_model(model)
                        constructor.assert_not_called()
                    else:
                        self.assertIs(
                            model_cls._create_python_model(model),
                            constructor.return_value,
                        )
                        constructor.assert_called_once()


if __name__ == "__main__":
    unittest.main()
