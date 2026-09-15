import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv4.block import Block
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.mega_front import (
    MEGA_MOE_FRONT_CAPACITY,
    MegaMoeFrontAdapter,
    _validate_extension_contract,
    moe_front_mode,
    moe_front_requested,
)


class _FakePlan:
    def __init__(self) -> None:
        self.calls = []
        self.closed = False

    def run_learned_out(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))
        normalized = args[8]
        post = args[14]
        comb = args[15]
        normalized[:2].fill_(3)
        post[:2].fill_(4)
        comb[:2].fill_(5)

    def close(self) -> None:
        self.closed = True


class _FakeStrategy:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.launches = []
        self._mega_buf = SimpleNamespace(
            num_max_tokens_per_rank=256,
            x=torch.empty((256, dim), dtype=torch.float8_e4m3fn),
            x_sf=torch.empty((256, max(dim // 128, 1)), dtype=torch.int32),
            shared_l1_acts_sf=torch.empty((1, 128), dtype=torch.int32),
            topk_idx=torch.empty((256, 6), dtype=torch.int64),
            topk_weights=torch.empty((256, 6), dtype=torch.float32),
        )

    def _block_m(self, tokens: int) -> int:
        self.launches.append(("block_m", tokens))
        return 16

    def forward_prepacked(self, tokens: int, device: torch.device) -> torch.Tensor:
        self.launches.append(("launch", tokens, device))
        return torch.full((tokens, self.dim), 7, dtype=torch.bfloat16)


class _FakeGate:
    hash = False
    route_scale = 2.5


class _FakeHC:
    hc_eps = 1.0e-6


class _FakeNorm:
    variance_epsilon = 1.0e-6


class _TensorContract:
    def __init__(self, *shape: int) -> None:
        self.shape = shape
        self.dtype = torch.bfloat16
        self.is_cuda = True

    def dim(self) -> int:
        return len(self.shape)

    def is_contiguous(self) -> bool:
        return True


def _fake_adapter(dim: int = 128) -> tuple[MegaMoeFrontAdapter, _FakePlan]:
    adapter = MegaMoeFrontAdapter.__new__(MegaMoeFrontAdapter)
    adapter.layer_id = 3
    adapter.dim = dim
    adapter.executor = _FakeStrategy(dim)
    adapter.gate = _FakeGate()
    adapter.ffn_hc = _FakeHC()
    adapter.ffn_norm = _FakeNorm()
    adapter.collapsed = torch.empty((256, dim), dtype=torch.bfloat16)
    adapter.collapse_ssq = torch.empty((256,), dtype=torch.float32)
    adapter.normalized_mix = torch.empty((256, 24), dtype=torch.float32)
    adapter.normalized = torch.empty((256, dim), dtype=torch.bfloat16)
    adapter.router_logits = torch.empty((256, 256), dtype=torch.float32)
    adapter.post = torch.empty((256, 4), dtype=torch.float32)
    adapter.comb = torch.empty((256, 4, 4), dtype=torch.float32)
    adapter.hc_base = torch.empty((24,), dtype=torch.float32)
    adapter.hc_fn = torch.empty((24, 4 * dim), dtype=torch.float32)
    adapter.hc_scale = torch.empty((3,), dtype=torch.float32)
    adapter.ffn_norm_weight = torch.empty((dim,), dtype=torch.bfloat16)
    adapter.router_weight = torch.empty((256, dim), dtype=torch.bfloat16)
    adapter.correction_bias = torch.empty((256,), dtype=torch.float32)
    adapter.tid2eid = None
    adapter._workspace = object()
    plan = _FakePlan()
    adapter.created_plans = []

    def create_plan(hidden, hc_fn, tokens, workspace):
        adapter.created_plans.append((hidden, hc_fn, tokens, workspace))
        return plan

    adapter._ops = SimpleNamespace(Dsv4MoeFrontPlan=create_plan)
    adapter._graph_plans = {}
    return adapter, plan


class MegaMoeFrontAdapterTest(unittest.TestCase):
    def test_front_is_attached_when_explicitly_enabled_for_mega_se(self) -> None:
        block = SimpleNamespace(
            ffn=SimpleNamespace(
                strategy_name="mega_moe_se",
                gate=SimpleNamespace(score_func="sqrtsoftplus"),
            ),
            ffn_hc="hc",
            ffn_norm="norm",
            _moe_front_adapter=None,
        )
        adapter = object()
        with mock.patch(
            "rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.mega_front.MegaMoeFrontAdapter",
            return_value=adapter,
        ) as adapter_cls:
            Block.enable_moe_front(block)

        self.assertIs(block._moe_front_adapter, adapter)
        adapter_cls.assert_called_once_with(
            block.ffn, "hc", "norm"
        )

    def test_front_is_not_attached_to_non_mega_strategy(self) -> None:
        block = SimpleNamespace(
            ffn=SimpleNamespace(strategy_name="local_loop"),
            ffn_hc="hc",
            ffn_norm="norm",
            _moe_front_adapter=None,
        )
        with mock.patch(
            "rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.mega_front.MegaMoeFrontAdapter"
        ) as adapter_cls:
            Block.enable_moe_front(block)

        self.assertIsNone(block._moe_front_adapter)
        adapter_cls.assert_not_called()

    def test_required_front_rejects_non_mega_se_strategy(self) -> None:
        block = SimpleNamespace(
            ffn=SimpleNamespace(strategy_name="mega_moe"),
            ffn_hc="hc",
            ffn_norm="norm",
            _moe_front_adapter=None,
        )

        with self.assertRaisesRegex(RuntimeError, "requires the mega_moe_se strategy"):
            Block.enable_moe_front(block, required=True)

    def test_front_is_not_attached_for_unsupported_score_func(self) -> None:
        block = SimpleNamespace(
            ffn=SimpleNamespace(
                strategy_name="mega_moe_se",
                gate=SimpleNamespace(score_func="softmax"),
            ),
            ffn_hc="hc",
            ffn_norm="norm",
            _moe_front_adapter=None,
        )
        with mock.patch(
            "rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.mega_front.MegaMoeFrontAdapter"
        ) as adapter_cls:
            Block.enable_moe_front(block)

        self.assertIsNone(block._moe_front_adapter)
        adapter_cls.assert_not_called()

    def test_required_front_rejects_unsupported_score_func(self) -> None:
        block = SimpleNamespace(
            ffn=SimpleNamespace(
                strategy_name="mega_moe_se",
                gate=SimpleNamespace(score_func="sigmoid"),
            ),
            ffn_hc="hc",
            ffn_norm="norm",
            _moe_front_adapter=None,
        )
        with self.assertRaisesRegex(RuntimeError, "score_func='sqrtsoftplus'"):
            Block.enable_moe_front(block, required=True)

    def test_front_requires_explicit_valid_environment_value(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(moe_front_requested())
        for value in ("1", "TRUE", "yes", "On", "auto"):
            with self.subTest(value=value), mock.patch.dict(
                "os.environ", {"DSV4_MEGA_MOE_FRONT": value}, clear=True
            ):
                self.assertTrue(moe_front_requested())
        for value in ("0", "false", "NO", "off", ""):
            with self.subTest(value=value), mock.patch.dict(
                "os.environ", {"DSV4_MEGA_MOE_FRONT": value}, clear=True
            ):
                self.assertFalse(moe_front_requested())
        with mock.patch.dict(
            "os.environ", {"DSV4_MEGA_MOE_FRONT": "auto"}, clear=True
        ):
            self.assertEqual(moe_front_mode(), "auto")
        with mock.patch.dict(
            "os.environ", {"DSV4_MEGA_MOE_FRONT": "1"}, clear=True
        ):
            self.assertEqual(moe_front_mode(), "required")
        with mock.patch.dict(
            "os.environ", {"DSV4_MEGA_MOE_FRONT": "maybe"}, clear=True
        ):
            with self.assertRaisesRegex(RuntimeError, "DSV4_MEGA_MOE_FRONT"):
                moe_front_requested()

    def test_plans_share_one_layer_workspace(self) -> None:
        adapter, _ = _fake_adapter()
        first_input = torch.empty((16, 4, adapter.dim), dtype=torch.bfloat16)
        second_input = torch.empty((128, 4, adapter.dim), dtype=torch.bfloat16)

        adapter._create_plan(first_input, 16)
        adapter._create_plan(second_input, 128)

        self.assertIs(adapter.created_plans[0][3], adapter._workspace)
        self.assertIs(adapter.created_plans[1][3], adapter._workspace)

    def test_graph_plans_are_cached_by_input_address(self) -> None:
        adapter, _ = _fake_adapter()
        first_input = torch.empty((16, 4, adapter.dim), dtype=torch.bfloat16)
        second_input = torch.empty((16, 4, adapter.dim), dtype=torch.bfloat16)
        created = []

        def create_plan(hidden, hc_fn, tokens, workspace):
            plan = _FakePlan()
            created.append((plan, hidden, hc_fn, tokens, workspace))
            return plan

        adapter._ops = SimpleNamespace(Dsv4MoeFrontPlan=create_plan)

        with mock.patch("torch.cuda.is_current_stream_capturing", return_value=True):
            first, first_temporary = adapter._plan_for(first_input, 16)
            again, again_temporary = adapter._plan_for(first_input, 16)
            second, second_temporary = adapter._plan_for(second_input, 16)

        self.assertIs(first, again)
        self.assertIsNot(second, first)
        self.assertFalse(first_temporary)
        self.assertFalse(again_temporary)
        self.assertFalse(second_temporary)
        self.assertEqual(len(created), 2)

    def test_front_support_is_bounded_by_extension_and_mega_buffer(self) -> None:
        adapter, _ = _fake_adapter()

        self.assertTrue(adapter.supports(_TensorContract(64, 2, 4, adapter.dim)))
        self.assertTrue(adapter.supports(_TensorContract(128, 2, 4, adapter.dim)))
        self.assertTrue(adapter.supports(_TensorContract(256, 1, 4, adapter.dim)))
        self.assertFalse(adapter.supports(_TensorContract(257, 1, 4, adapter.dim)))
        self.assertTrue(adapter.supports(_TensorContract(43, 3, 4, adapter.dim)))
        self.assertFalse(adapter.supports(_TensorContract(43, 6, 4, adapter.dim)))

        adapter.executor._mega_buf.num_max_tokens_per_rank = 32
        self.assertTrue(adapter.supports(_TensorContract(16, 2, 4, adapter.dim)))
        self.assertFalse(adapter.supports(_TensorContract(17, 2, 4, adapter.dim)))

        adapter.gate.hash = True
        hash_residual = _TensorContract(16, 2, 4, adapter.dim)
        self.assertTrue(
            adapter.supports(hash_residual, torch.empty(32, dtype=torch.int32))
        )
        self.assertFalse(
            adapter.supports(hash_residual, torch.empty(32, dtype=torch.int64))
        )

    def test_validates_v3_sm103_extension_contract(self) -> None:
        geometry = {
            "abi_version": 1,
            "kernel_contract_version": 3,
            "hidden": 4096,
            "hc_mult": 4,
            "hc_width": 24,
            "experts": 256,
            "topk": 6,
            "max_m": MEGA_MOE_FRONT_CAPACITY,
            "scale_cols": 32,
            "collapse_ssq_bits": 32,
            "hash_input_id_bits": 32,
        }
        valid_build_info = {
            "source_commit": "37c78f10b54fd37cab72d90f37cb89cd27e67e7e",
            "source_sha256": "7" * 64,
            "deepgemm_commit": "559d79fb6994a58b8a15b4b93bf13ccc16edf247",
            "cutlass_commit": "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8",
            "target_arches": "sm_100a,sm_103a",
            "production_arch": "sm_100a,sm_103a",
            "kernel_count": 4,
        }
        ops = SimpleNamespace(
            geometry_moe_front=lambda _hidden: geometry,
            build_info_moe_front=lambda: valid_build_info,
        )
        with mock.patch("torch.cuda.get_device_capability", return_value=(10, 3)):
            self.assertEqual(
                _validate_extension_contract(
                    ops, 4096, 256, 6, torch.device("cuda:0")
                ),
                geometry,
            )

        ops.geometry_moe_front = lambda _hidden: dict(
            geometry, kernel_contract_version=2
        )
        with mock.patch("torch.cuda.get_device_capability", return_value=(10, 3)):
            with self.assertRaisesRegex(RuntimeError, "geometry mismatch"):
                _validate_extension_contract(
                    ops, 4096, 256, 6, torch.device("cuda:0")
                )

        for missing_field in ("scale_cols", "collapse_ssq_bits"):
            with self.subTest(missing_field=missing_field):
                incomplete = dict(geometry)
                incomplete.pop(missing_field)
                ops.geometry_moe_front = lambda _hidden, value=incomplete: value
                with mock.patch(
                    "torch.cuda.get_device_capability", return_value=(10, 3)
                ):
                    with self.assertRaisesRegex(RuntimeError, "geometry mismatch"):
                        _validate_extension_contract(
                            ops, 4096, 256, 6, torch.device("cuda:0")
                        )

        ops.geometry_moe_front = lambda _hidden: None
        with self.assertRaisesRegex(RuntimeError, "must be a mapping"):
            _validate_extension_contract(
                ops, 4096, 256, 6, torch.device("cuda:0")
            )

        ops.geometry_moe_front = lambda _hidden: geometry
        for field in ("deepgemm_commit", "cutlass_commit"):
            with self.subTest(field=field):
                invalid_build_info = dict(valid_build_info)
                invalid_build_info.pop(field)
                ops.build_info_moe_front = (
                    lambda value=invalid_build_info: value
                )
                with mock.patch(
                    "torch.cuda.get_device_capability", return_value=(10, 3)
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, f"invalid dependency identity {field}"
                    ):
                        _validate_extension_contract(
                            ops, 4096, 256, 6, torch.device("cuda:0")
                        )

    def test_learned_front_stages_and_launches_prepacked_mega(self) -> None:
        adapter, plan = _fake_adapter()
        residual = torch.arange(2 * 4 * adapter.dim, dtype=torch.float32)
        residual = residual.to(torch.bfloat16).view(2, 1, 4, adapter.dim)
        input_ids = torch.tensor([[11], [12]], dtype=torch.int64)

        y, normalized, post, comb = adapter.forward(residual, input_ids)

        self.assertEqual(len(adapter.created_plans), 1)
        self.assertEqual(adapter.created_plans[0][0].data_ptr(), residual.data_ptr())
        self.assertEqual(tuple(y.shape), (2, 1, adapter.dim))
        self.assertEqual(tuple(normalized.shape), (2, 1, adapter.dim))
        self.assertEqual(tuple(post.shape), (2, 1, 4, 1))
        self.assertEqual(tuple(comb.shape), (2, 1, 4, 4))
        self.assertTrue(torch.all(y == 7))
        self.assertTrue(torch.all(normalized == 3))
        self.assertTrue(torch.all(post == 4))
        self.assertTrue(torch.all(comb == 5))
        self.assertEqual(adapter.executor.launches[0], ("block_m", 2))
        self.assertEqual(adapter.executor.launches[1][:2], ("launch", 2))
        self.assertEqual(len(plan.calls), 1)
        args, kwargs = plan.calls[0]
        self.assertEqual(args[16], 16)
        self.assertEqual(tuple(args[9].shape), (2, adapter.dim))
        self.assertEqual(tuple(args[10].shape), (2, 1))
        self.assertEqual(tuple(args[12].shape), (2, 6))
        self.assertEqual(tuple(args[13].shape), (2, 6))
        self.assertIs(kwargs["router_logits"], adapter.router_logits)
        self.assertTrue(plan.closed)

    def test_empty_rank_skips_front_and_enters_mega_collective(self) -> None:
        adapter, plan = _fake_adapter()
        residual = torch.empty((0, 1, 4, adapter.dim), dtype=torch.bfloat16)
        input_ids = torch.empty((0, 1), dtype=torch.int64)

        y, normalized, post, comb = adapter.forward(residual, input_ids)

        self.assertEqual(tuple(y.shape), (0, 1, adapter.dim))
        self.assertEqual(tuple(normalized.shape), (0, 1, adapter.dim))
        self.assertEqual(tuple(post.shape), (0, 1, 4, 1))
        self.assertEqual(tuple(comb.shape), (0, 1, 4, 4))
        self.assertEqual(adapter.executor.launches[0][:2], ("launch", 0))
        self.assertEqual(plan.calls, [])


if __name__ == "__main__":
    unittest.main()
