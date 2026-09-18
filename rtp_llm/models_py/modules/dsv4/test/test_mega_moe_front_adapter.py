import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4.block import Block
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se import (
    MegaMoeSEExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.mega_front import (
    MEGA_MOE_FRONT_CAPACITY,
    MegaMoeFrontAdapter,
    _load_and_validate_extension,
    _validate_extension_contract,
    moe_front_mode,
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

    def run_hash_out(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))
        normalized = args[9]
        post = args[16]
        comb = args[17]
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
    def __init__(self, *shape: int, device: str = "cuda:0") -> None:
        self.shape = shape
        self.dtype = torch.bfloat16
        self.is_cuda = True
        self.device = torch.device(device)

    def dim(self) -> int:
        return len(self.shape)

    def is_contiguous(self) -> bool:
        return True


class _InputIdsContract:
    def __init__(
        self,
        tokens: int,
        *,
        dtype: torch.dtype = torch.int32,
        device: str = "cuda:0",
        contiguous: bool = True,
    ) -> None:
        self._tokens = tokens
        self.dtype = dtype
        self.device = torch.device(device)
        self.is_cuda = self.device.type == "cuda"
        self._contiguous = contiguous

    def numel(self) -> int:
        return self._tokens

    def is_contiguous(self) -> bool:
        return self._contiguous


def _reference_prefix(adapter, residual, input_ids):
    tokens = residual.shape[0]
    flat = residual.view(tokens, -1).float()
    inv_rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1.0e-6)
    mix = F.linear(flat, adapter.hc_fn.float()) * inv_rms
    pre = (
        torch.sigmoid(mix[:, :4] * adapter.hc_scale[0] + adapter.hc_base[:4])
        + adapter.ffn_hc.hc_eps
    )
    collapsed = torch.sum(pre.unsqueeze(-1) * residual.float(), dim=1).to(
        torch.bfloat16
    )
    ssq = collapsed.float().square().sum(-1)
    normalized = (
        collapsed.float()
        * torch.rsqrt(ssq[:, None] / adapter.dim + adapter.ffn_norm.variance_epsilon)
        * adapter.ffn_norm_weight.float()
    ).to(torch.bfloat16)

    post = 2.0 * torch.sigmoid(mix[:, 4:8] * adapter.hc_scale[1] + adapter.hc_base[4:8])
    comb = mix[:, 8:].view(tokens, 4, 4)
    comb = torch.softmax(
        comb * adapter.hc_scale[2] + adapter.hc_base[8:].view(4, 4), dim=-1
    )
    comb = comb + adapter.ffn_hc.hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + adapter.ffn_hc.hc_eps)
    for _ in range(19):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + adapter.ffn_hc.hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + adapter.ffn_hc.hc_eps)

    logits = F.linear(normalized.float(), adapter.router_weight.float())
    scores = F.softplus(logits).sqrt()
    if adapter.gate.hash:
        topk_ids = adapter.tid2eid[input_ids.view(-1)].to(torch.int64)
    else:
        topk_ids = (scores + adapter.correction_bias).topk(6, dim=-1).indices
    selected = scores.gather(1, topk_ids)
    topk_weights = selected / selected.sum(-1, keepdim=True) * adapter.gate.route_scale
    return normalized, post, comb, topk_ids, topk_weights


class _ReferencePlan(_FakePlan):
    def __init__(self, adapter, residual, tokens: int) -> None:
        super().__init__()
        self.adapter = adapter
        self.residual = residual
        self.tokens = tokens

    def _write_reference(self, input_ids, args, *, hash_mode: bool) -> None:
        normalized, post, comb, topk_ids, topk_weights = _reference_prefix(
            self.adapter, self.residual[: self.tokens], input_ids
        )
        if hash_mode:
            normalized_out, topk_ids_out, topk_weights_out = args[9], args[14], args[15]
            post_out, comb_out = args[16], args[17]
        else:
            normalized_out, topk_ids_out, topk_weights_out = args[8], args[12], args[13]
            post_out, comb_out = args[14], args[15]
        normalized_out[: self.tokens].copy_(normalized)
        topk_ids_out[: self.tokens].copy_(topk_ids)
        topk_weights_out[: self.tokens].copy_(topk_weights)
        post_out[: self.tokens].copy_(post)
        comb_out[: self.tokens].copy_(comb)

    def run_learned_out(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))
        input_ids = torch.zeros(self.tokens, dtype=torch.int64)
        self._write_reference(input_ids, args, hash_mode=False)

    def run_hash_out(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))
        self._write_reference(args[4], args, hash_mode=True)


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
        adapter_cls.assert_called_once_with(block.ffn, "hc", "norm")

    def test_repeated_front_attach_closes_the_previous_adapter(self) -> None:
        old_adapter = mock.Mock()
        new_adapter = object()
        block = SimpleNamespace(
            ffn=SimpleNamespace(
                strategy_name="mega_moe_se",
                gate=SimpleNamespace(score_func="sqrtsoftplus"),
            ),
            ffn_hc="hc",
            ffn_norm="norm",
            _moe_front_adapter=old_adapter,
        )
        with mock.patch(
            "rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.mega_front.MegaMoeFrontAdapter",
            return_value=new_adapter,
        ):
            Block.enable_moe_front(block, required=True)

        old_adapter.close.assert_called_once_with()
        self.assertIs(block._moe_front_adapter, new_adapter)

    def test_front_is_not_attached_to_non_mega_strategy(self) -> None:
        block = SimpleNamespace(
            ffn=SimpleNamespace(strategy_name="local_loop"),
            ffn_hc="hc",
            ffn_norm="norm",
            _moe_front_adapter=None,
        )
        with self.assertLogs(level="INFO") as logs:
            with mock.patch(
                "rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.mega_front.MegaMoeFrontAdapter"
            ) as adapter_cls:
                Block.enable_moe_front(block)

        self.assertIsNone(block._moe_front_adapter)
        adapter_cls.assert_not_called()
        self.assertIn("unsupported strategy='local_loop'", "\n".join(logs.output))

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

    def test_auto_mode_falls_back_for_expected_adapter_validation_errors(self) -> None:
        for error in (ImportError, AttributeError, TypeError, ValueError, RuntimeError):
            block = SimpleNamespace(
                ffn=SimpleNamespace(
                    strategy_name="mega_moe_se",
                    gate=SimpleNamespace(score_func="sqrtsoftplus"),
                ),
                ffn_hc="hc",
                ffn_norm="norm",
                _moe_front_adapter=None,
            )
            with self.subTest(error=error.__name__), mock.patch(
                "rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.mega_front.MegaMoeFrontAdapter",
                side_effect=error("not available"),
            ):
                Block.enable_moe_front(block)
            self.assertIsNone(block._moe_front_adapter)

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
            self.assertEqual(moe_front_mode(), "off")
        for value in ("1", "TRUE", "yes", "On", "auto"):
            with self.subTest(value=value), mock.patch.dict(
                "os.environ", {"DSV4_MEGA_MOE_FRONT": value}, clear=True
            ):
                self.assertNotEqual(moe_front_mode(), "off")
        for value in ("0", "false", "NO", "off", ""):
            with self.subTest(value=value), mock.patch.dict(
                "os.environ", {"DSV4_MEGA_MOE_FRONT": value}, clear=True
            ):
                self.assertEqual(moe_front_mode(), "off")
        with mock.patch.dict("os.environ", {"DSV4_MEGA_MOE_FRONT": "auto"}, clear=True):
            self.assertEqual(moe_front_mode(), "auto")
        with mock.patch.dict("os.environ", {"DSV4_MEGA_MOE_FRONT": "1"}, clear=True):
            self.assertEqual(moe_front_mode(), "required")
        with mock.patch.dict(
            "os.environ", {"DSV4_MEGA_MOE_FRONT": "maybe"}, clear=True
        ):
            with self.assertRaisesRegex(RuntimeError, "DSV4_MEGA_MOE_FRONT"):
                moe_front_mode()

    def test_front_switch_is_independent_from_attention_switches(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "DSV4_MEGA_MOE_FRONT": "1",
                "DSV4_MEGA_CSA": "0",
                "DSV4_MEGA_HCA": "0",
            },
            clear=True,
        ):
            self.assertEqual(moe_front_mode(), "required")

        with mock.patch.dict(
            "os.environ",
            {
                "DSV4_MEGA_MOE_FRONT": "0",
                "DSV4_MEGA_CSA": "1",
                "DSV4_MEGA_HCA": "1",
            },
            clear=True,
        ):
            self.assertEqual(moe_front_mode(), "off")

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

        adapter.close()
        self.assertEqual(adapter._graph_plans, {})
        self.assertTrue(first.closed)
        self.assertTrue(second.closed)

    def test_disable_front_closes_plans(self) -> None:
        adapter, plan = _fake_adapter()
        adapter._graph_plans[(16, 1234)] = plan
        block = SimpleNamespace(
            _moe_front_adapter=adapter,
        )

        Block.disable_moe_front(block)

        self.assertIsNone(block._moe_front_adapter)
        self.assertTrue(plan.closed)

    def test_front_support_is_bounded_by_extension_and_mega_buffer(self) -> None:
        adapter, _ = _fake_adapter()

        for tokens in (1, 128, 129, 256):
            with self.subTest(tokens=tokens):
                self.assertTrue(
                    adapter.supports(
                        _TensorContract(tokens, 1, 4, adapter.dim),
                        torch.empty(tokens, dtype=torch.int64),
                    )
                )
        self.assertFalse(
            adapter.supports(
                _TensorContract(257, 1, 4, adapter.dim),
                torch.empty(257, dtype=torch.int64),
            )
        )
        self.assertTrue(
            adapter.supports(
                _TensorContract(43, 3, 4, adapter.dim),
                torch.empty(129, dtype=torch.int64),
            )
        )
        self.assertFalse(
            adapter.supports(
                _TensorContract(43, 6, 4, adapter.dim),
                torch.empty(258, dtype=torch.int64),
            )
        )
        self.assertFalse(adapter.supports(_TensorContract(16, 2, 4, adapter.dim)))

        adapter.executor._mega_buf.num_max_tokens_per_rank = 32
        self.assertTrue(
            adapter.supports(
                _TensorContract(16, 2, 4, adapter.dim),
                torch.empty(32, dtype=torch.int64),
            )
        )
        self.assertFalse(
            adapter.supports(
                _TensorContract(17, 2, 4, adapter.dim),
                torch.empty(34, dtype=torch.int64),
            )
        )

        adapter.gate.hash = True
        hash_residual = _TensorContract(16, 2, 4, adapter.dim)
        self.assertTrue(
            adapter.supports(hash_residual, _InputIdsContract(32))
        )
        self.assertFalse(
            adapter.supports(
                hash_residual, _InputIdsContract(32, dtype=torch.int64)
            )
        )
        self.assertEqual(
            adapter.unsupported_reason(
                hash_residual, _InputIdsContract(32, dtype=torch.int64)
            ),
            "HashMoE input_ids dtype is torch.int64, expected torch.int32",
        )
        self.assertFalse(
            adapter.supports(
                hash_residual, _InputIdsContract(32, device="cuda:1")
            )
        )
        self.assertFalse(
            adapter.supports(hash_residual, _InputIdsContract(32, device="cpu"))
        )

        adapter.gate.hash = False
        noncontiguous_ids = torch.empty((2, 16), dtype=torch.int64).transpose(0, 1)
        self.assertTrue(adapter.supports(hash_residual, noncontiguous_ids))

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
                _validate_extension_contract(ops, 4096, 256, 6, torch.device("cuda:0")),
                geometry,
            )

        short_commit_build_info = dict(valid_build_info, source_commit="37c78f10")
        ops.build_info_moe_front = lambda: short_commit_build_info
        with mock.patch("torch.cuda.get_device_capability", return_value=(10, 3)):
            self.assertEqual(
                _validate_extension_contract(ops, 4096, 256, 6, torch.device("cuda:0")),
                geometry,
            )
        ops.build_info_moe_front = lambda: valid_build_info

        ops.geometry_moe_front = lambda _hidden: dict(
            geometry, kernel_contract_version=2
        )
        with mock.patch("torch.cuda.get_device_capability", return_value=(10, 3)):
            with self.assertRaisesRegex(RuntimeError, "geometry mismatch"):
                _validate_extension_contract(ops, 4096, 256, 6, torch.device("cuda:0"))

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
            _validate_extension_contract(ops, 4096, 256, 6, torch.device("cuda:0"))

        ops.geometry_moe_front = lambda _hidden: geometry
        for field in ("deepgemm_commit", "cutlass_commit"):
            with self.subTest(field=field):
                invalid_build_info = dict(valid_build_info)
                invalid_build_info.pop(field)
                ops.build_info_moe_front = lambda value=invalid_build_info: value
                with mock.patch(
                    "torch.cuda.get_device_capability", return_value=(10, 3)
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, f"invalid dependency identity {field}"
                    ):
                        _validate_extension_contract(
                            ops, 4096, 256, 6, torch.device("cuda:0")
                        )

    def test_rejects_hidden_size_without_fp8_scale_alignment(self) -> None:
        ops = SimpleNamespace(geometry_moe_front=mock.Mock())

        with self.assertRaisesRegex(RuntimeError, "multiple of 128"):
            _validate_extension_contract(ops, 4100, 256, 6, torch.device("cuda:0"))

        ops.geometry_moe_front.assert_not_called()

    def test_extension_contract_is_validated_once_per_device_geometry(self) -> None:
        ops = object()
        geometry = {"max_m": MEGA_MOE_FRONT_CAPACITY}
        extension_package = SimpleNamespace(dsv4_moe_front=ops)
        validation_target = (
            "rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe."
            "mega_front._validate_extension_contract"
        )
        _load_and_validate_extension.cache_clear()
        try:
            with mock.patch.dict("sys.modules", {"rtp_kernel": extension_package}):
                with mock.patch(validation_target, return_value=geometry) as validate:
                    first = _load_and_validate_extension(
                        4096, 256, 6, torch.device("cuda:0")
                    )
                    second = _load_and_validate_extension(
                        4096, 256, 6, torch.device("cuda:0")
                    )
        finally:
            _load_and_validate_extension.cache_clear()

        self.assertIs(first, second)
        validate.assert_called_once_with(
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

    def test_learned_front_avoids_logits_buffer_above_low_m_specialization(
        self,
    ) -> None:
        adapter, plan = _fake_adapter()
        residual = torch.zeros((10, 1, 4, adapter.dim), dtype=torch.bfloat16)
        input_ids = torch.arange(10, dtype=torch.int64).view(10, 1)

        adapter.forward(residual, input_ids)

        self.assertEqual(len(plan.calls), 1)
        _, kwargs = plan.calls[0]
        self.assertIsNone(kwargs["router_logits"])

    def test_front_prefix_matches_unfused_reference_for_learned_and_hash(self) -> None:
        torch.manual_seed(20260916)
        tokens = 4
        dim = 128
        residual = torch.randn(tokens, 1, 4, dim).to(torch.bfloat16)

        for hash_mode in (False, True):
            with self.subTest(hash_mode=hash_mode):
                adapter, _ = _fake_adapter(dim)
                adapter.gate.hash = hash_mode
                adapter.hc_fn.normal_(mean=0.0, std=0.02)
                adapter.hc_base.normal_(mean=0.0, std=0.05)
                adapter.hc_scale.copy_(torch.tensor([0.25, 0.5, 0.75]))
                adapter.ffn_norm_weight.normal_(mean=1.0, std=0.02)
                adapter.router_weight.normal_(mean=0.0, std=0.03)
                adapter.correction_bias = torch.randn(256) * 0.01
                input_ids = torch.tensor([[2], [7], [11], [19]], dtype=torch.int64)
                if hash_mode:
                    adapter.tid2eid = torch.randint(0, 256, (32, 6), dtype=torch.int32)
                    adapter.correction_bias = None
                    input_ids = input_ids.to(torch.int32)

                created = []

                def create_plan(hidden, hc_fn, plan_tokens, workspace):
                    plan = _ReferencePlan(adapter, hidden, plan_tokens)
                    created.append(plan)
                    return plan

                adapter._ops = SimpleNamespace(Dsv4MoeFrontPlan=create_plan)
                expected = _reference_prefix(
                    adapter, residual.view(tokens, 4, dim), input_ids
                )

                y, normalized, post, comb = adapter.forward(residual, input_ids)

                torch.testing.assert_close(
                    normalized.view(tokens, dim), expected[0], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    post.view(tokens, 4), expected[1], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    comb.view(tokens, 4, 4), expected[2], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    adapter.executor._mega_buf.topk_idx[:tokens],
                    expected[3],
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    adapter.executor._mega_buf.topk_weights[:tokens],
                    expected[4],
                    rtol=1.0e-6,
                    atol=1.0e-6,
                )

                front_final = post.to(y.dtype) * y.unsqueeze(-2) + torch.matmul(
                    comb.to(residual.dtype).transpose(-1, -2), residual
                )
                expected_post = expected[1].view(tokens, 1, 4, 1)
                expected_comb = expected[2].view(tokens, 1, 4, 4)
                ordinary_final = expected_post.to(y.dtype) * y.unsqueeze(
                    -2
                ) + torch.matmul(
                    expected_comb.to(residual.dtype).transpose(-1, -2), residual
                )
                torch.testing.assert_close(front_final, ordinary_final)
                self.assertTrue(created[0].closed)

    def test_hash_front_passes_int32_ids_to_four_kernel_plan(self) -> None:
        adapter, plan = _fake_adapter()
        adapter.gate.hash = True
        adapter.correction_bias = None
        adapter.tid2eid = torch.arange(256, dtype=torch.int32)
        residual = torch.arange(2 * 4 * adapter.dim, dtype=torch.float32)
        residual = residual.to(torch.bfloat16).view(2, 1, 4, adapter.dim)
        input_ids = torch.tensor([[11], [12]], dtype=torch.int32)

        y, normalized, post, comb = adapter.forward(residual, input_ids)

        self.assertEqual(tuple(y.shape), (2, 1, adapter.dim))
        self.assertTrue(torch.all(normalized == 3))
        self.assertTrue(torch.all(post == 4))
        self.assertTrue(torch.all(comb == 5))
        self.assertEqual(len(plan.calls), 1)
        args, _ = plan.calls[0]
        self.assertEqual(args[4].data_ptr(), input_ids.data_ptr())
        self.assertEqual(args[4].dtype, torch.int32)
        self.assertIs(args[5], adapter.tid2eid)
        self.assertEqual(args[18], 16)
        self.assertTrue(plan.closed)

    def test_decode_fallback_and_disabled_front_use_original_moe_path(self) -> None:
        dim = 8
        residual = torch.ones((257, 1, 4, dim), dtype=torch.bfloat16)
        input_ids = torch.arange(257, dtype=torch.int64).view(257, 1)
        collapsed = residual.mean(dim=-2)

        attn_hc = SimpleNamespace(
            pre=mock.Mock(return_value=(collapsed, object(), object())),
            post=mock.Mock(return_value=residual),
        )
        ffn_hc = SimpleNamespace(
            pre=mock.Mock(return_value=(collapsed, object(), object())),
            post=mock.Mock(return_value=residual),
        )
        adapter = SimpleNamespace(
            supports=mock.Mock(return_value=False),
            unsupported_reason=mock.Mock(
                return_value="decode token count 257 exceeds capacity 256"
            ),
            forward=mock.Mock(side_effect=AssertionError("front must not run")),
        )
        ffn = mock.Mock(return_value=collapsed)
        block = SimpleNamespace(
            layer_id=3,
            attn_hc=attn_hc,
            attn_norm=mock.Mock(side_effect=lambda value: value),
            attn=None,
            ffn_hc=ffn_hc,
            ffn_norm=mock.Mock(side_effect=lambda value: value),
            ffn=ffn,
            _moe_front_adapter=adapter,
            _moe_front_fallback_logged=False,
        )

        with self.assertLogs(level="WARNING") as logs:
            with mock.patch(
                "rtp_llm.models_py.modules.dsv4._record_tensor.should_record_layer",
                return_value=False,
            ):
                output = Block.forward_decode(
                    block,
                    residual,
                    SimpleNamespace(),
                    input_ids,
                    attn_fn=lambda value: value,
                )

        self.assertIs(output, residual)
        adapter.supports.assert_called_once_with(residual, input_ids)
        adapter.unsupported_reason.assert_called_once_with(residual, input_ids)
        adapter.forward.assert_not_called()
        self.assertIn(
            "decode token count 257 exceeds capacity 256", "\n".join(logs.output)
        )
        ffn_hc.pre.assert_called_once()
        ffn.assert_called_once()
        ffn_input, ffn_input_ids = ffn.call_args.args
        self.assertEqual(ffn_input.data_ptr(), collapsed.data_ptr())
        self.assertIs(ffn_input_ids, input_ids)

        block._moe_front_adapter = None
        ffn_hc.pre.reset_mock()
        ffn.reset_mock()
        with mock.patch(
            "rtp_llm.models_py.modules.dsv4._record_tensor.should_record_layer",
            return_value=False,
        ):
            output = Block.forward_decode(
                block,
                residual,
                SimpleNamespace(),
                input_ids,
                attn_fn=lambda value: value,
            )

        self.assertIs(output, residual)
        ffn_hc.pre.assert_called_once()
        ffn.assert_called_once()

    def test_required_attach_preserves_runtime_capacity_fallback(self) -> None:
        dim = 8
        residual = torch.ones((257, 1, 4, dim), dtype=torch.bfloat16)
        input_ids = torch.arange(257, dtype=torch.int64).view(257, 1)
        collapsed = residual.mean(dim=-2)
        block = SimpleNamespace(
            layer_id=3,
            attn_hc=SimpleNamespace(
                pre=mock.Mock(return_value=(collapsed, object(), object())),
                post=mock.Mock(return_value=residual),
            ),
            attn_norm=mock.Mock(side_effect=lambda value: value),
            attn=None,
            ffn_hc=SimpleNamespace(
                pre=mock.Mock(return_value=(collapsed, object(), object())),
                post=mock.Mock(return_value=residual),
            ),
            ffn_norm=mock.Mock(side_effect=lambda value: value),
            ffn=mock.Mock(return_value=collapsed),
            _moe_front_adapter=SimpleNamespace(
                supports=mock.Mock(return_value=False),
                unsupported_reason=mock.Mock(return_value="capacity mismatch"),
            ),
            _moe_front_fallback_logged=False,
        )

        with mock.patch(
            "rtp_llm.models_py.modules.dsv4._record_tensor.should_record_layer",
            return_value=False,
        ):
            output = Block.forward_decode(
                block,
                residual,
                SimpleNamespace(),
                input_ids,
                attn_fn=lambda value: value,
            )

        self.assertIs(output, residual)
        block.ffn.assert_called_once()

    def test_front_records_ffn_input_before_routed_expert_launch(self) -> None:
        adapter, _ = _fake_adapter()
        residual = torch.ones((2, 1, 4, adapter.dim), dtype=torch.bfloat16)
        input_ids = torch.ones((2, 1), dtype=torch.int64)
        events = []

        def launch(tokens: int, device: torch.device) -> torch.Tensor:
            events.append("ffn_launch")
            return torch.empty((tokens, adapter.dim), dtype=torch.bfloat16)

        adapter.executor.forward_prepacked = launch
        adapter.forward(
            residual,
            input_ids,
            ffn_input_observer=lambda _value: events.append("ffn_in"),
        )

        self.assertLess(events.index("ffn_in"), events.index("ffn_launch"))

    def test_fallback_records_ffn_input_before_launch(self) -> None:
        dim = 8
        residual = torch.ones((2, 1, 4, dim), dtype=torch.bfloat16)
        collapsed = residual.mean(dim=-2)
        events = []

        def run_ffn(value, input_ids, *, is_decode_forward):
            events.append("ffn_launch")
            return value

        block = SimpleNamespace(
            layer_id=3,
            attn_hc=SimpleNamespace(
                pre=mock.Mock(return_value=(collapsed, object(), object())),
                post=mock.Mock(return_value=residual),
            ),
            attn_norm=mock.Mock(side_effect=lambda value: value),
            attn=None,
            ffn_hc=SimpleNamespace(
                pre=mock.Mock(return_value=(collapsed, object(), object())),
                post=mock.Mock(return_value=residual),
            ),
            ffn_norm=mock.Mock(side_effect=lambda value: value),
            ffn=mock.Mock(side_effect=run_ffn),
            _moe_front_adapter=None,
            _moe_front_fallback_logged=False,
        )

        def record(_level, name, _tensor):
            events.append(name)

        with mock.patch(
            "rtp_llm.models_py.modules.dsv4._record_tensor.should_record_layer",
            return_value=True,
        ), mock.patch(
            "rtp_llm.models_py.modules.dsv4._record_tensor.record_if_level",
            side_effect=record,
        ):
            Block.forward_decode(
                block,
                residual,
                SimpleNamespace(),
                torch.ones((2, 1), dtype=torch.int64),
                attn_fn=lambda value: value,
            )

        self.assertLess(events.index("L03_decode_ffn_in"), events.index("ffn_launch"))

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

    def test_real_prepacked_executor_entry_handles_nonempty_and_empty_ranks(
        self,
    ) -> None:
        executor = MegaMoeSEExecutor.__new__(MegaMoeSEExecutor)
        executor._mega_buf = SimpleNamespace(num_max_tokens_per_rank=8)
        executor._mega_y = torch.arange(8 * 4, dtype=torch.bfloat16).view(8, 4)
        executor._launch = mock.Mock()
        device = torch.device("cpu")

        nonempty = executor.forward_prepacked(3, device)
        empty = executor.forward_prepacked(0, device)

        self.assertEqual(tuple(nonempty.shape), (3, 4))
        self.assertEqual(tuple(empty.shape), (0, 4))
        self.assertEqual(
            executor._launch.call_args_list,
            [mock.call(nonempty, 3, device), mock.call(empty, 0, device)],
        )


if __name__ == "__main__":
    unittest.main()
