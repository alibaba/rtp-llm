import unittest

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_dispatch_args import (
    get_low_latency_dispatch_quant_args,
)


class DeepEpDispatchArgsTest(unittest.TestCase):
    def test_quant_dispatch_args(self) -> None:
        cases = [
            (False, True, True, True, {}),
            (True, True, False, True, {"round_scale": True, "use_ue8m0": True}),
            (True, True, False, False, {}),
            (True, False, True, True, {"pertoken_quant": True}),
            (True, True, True, True, {"round_scale": True, "use_ue8m0": True}),
        ]
        for use_fp8, block_quant, per_token, use_e8m0, expected in cases:
            with self.subTest(
                use_fp8=use_fp8,
                block_quant=block_quant,
                per_token=per_token,
                use_e8m0=use_e8m0,
            ):
                self.assertEqual(
                    get_low_latency_dispatch_quant_args(
                        use_fp8=use_fp8,
                        is_block_quantized=block_quant,
                        is_per_act_token=per_token,
                        use_e8m0=use_e8m0,
                    ),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
