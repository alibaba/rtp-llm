import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

CAUSAL_CONV1D_PATH = Path(__file__).resolve().parents[1] / "causal_conv1d.py"


def _source_tree():
    return ast.parse(
        CAUSAL_CONV1D_PATH.read_text(encoding="utf-8"),
        filename=str(CAUSAL_CONV1D_PATH),
    )


class TestCausalConv1dWithoutRuntimeDependencies(unittest.TestCase):
    def test_module_is_valid_python(self):
        compile(_source_tree(), str(CAUSAL_CONV1D_PATH), "exec")

    def test_accelerator_implementations_are_lazy_imports(self):
        eager_imports = [
            node
            for node in _source_tree().body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        imported_modules = []
        for node in eager_imports:
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            else:
                imported_modules.append(node.module or "")

        self.assertFalse(
            any(
                "triton" in module or "fla_npu" in module for module in imported_modules
            ),
            imported_modules,
        )

    def test_public_api_is_present(self):
        public_names = {
            node.name
            for node in _source_tree().body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        }
        self.assertTrue(
            {
                "CausalConv1dMetadata",
                "prepare_causal_conv1d_metadata",
                "causal_conv1d_fn",
                "causal_conv1d_update",
            }.issubset(public_names)
        )


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this test environment")
class TestCausalConv1dNpuAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module_name = "ascendc_causal_conv1d_test"
        spec = importlib.util.spec_from_file_location(module_name, CAUSAL_CONV1D_PATH)
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = cls.module
        spec.loader.exec_module(cls.module)

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("ascendc_causal_conv1d_test", None)

    def test_prefill_layout_kwargs_and_cross_block_cache_scatter(self):
        # GPU-facing layout is (D, T); the AscendC fn entry receives (T, D)
        # with device-tensor metadata and (batch, state_len, dim) states.
        x = torch.tensor(
            [[10, 20, 30, 40, 50], [11, 21, 31, 41, 51]],
            dtype=torch.float16,
        )
        weight = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        conv_states = torch.zeros(6, 2, 3, dtype=torch.float32)
        conv_states[3] = torch.tensor(
            [[100, 101, 102], [200, 201, 202]], dtype=torch.float32
        )
        query_start_loc = torch.tensor([0, 5], dtype=torch.int32)
        prefix_lengths = torch.tensor([2], dtype=torch.int32)
        block_map = torch.tensor([[3, 4]], dtype=torch.int32)
        calls = []

        def fake_npu_causal_conv1d_fn(**kwargs):
            calls.append(
                {
                    key: value.clone() if isinstance(value, torch.Tensor) else value
                    for key, value in kwargs.items()
                }
            )
            return kwargs["x"]

        with mock.patch.object(
            self.module,
            "_load_npu_causal_conv1d_fn",
            return_value=fake_npu_causal_conv1d_fn,
        ):
            output = self.module.causal_conv1d_fn(
                x=x,
                weight=weight,
                bias=None,
                conv_states=conv_states,
                query_start_loc=query_start_loc,
                block_map=block_map,
                prefix_lengths=prefix_lengths,
                seq_size_per_block=4,
                activation="swish",
            )

        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(tuple(call["x"].shape), (5, 2))
        self.assertEqual(tuple(call["weight"].shape), (4, 2))
        # FLA-NPU states layout: (batch, state_len, dim), gathered from page 3
        self.assertEqual(tuple(call["conv_states"].shape), (1, 3, 2))
        torch.testing.assert_close(
            call["conv_states"][0],
            torch.tensor([[100, 200], [101, 201], [102, 202]], dtype=torch.float32),
        )
        # metadata passes as int32 device tensors / str activation
        self.assertEqual(tuple(call["query_start_loc"].shape), (2,))
        self.assertEqual(call["query_start_loc"].dtype, torch.int32)
        self.assertEqual(call["query_start_loc"].tolist(), [0, 5])
        self.assertEqual(call["has_initial_state"].dtype, torch.int32)
        self.assertEqual(call["has_initial_state"].tolist(), [1])
        self.assertEqual(call["activation"], "silu")
        self.assertEqual(output.dtype, x.dtype)
        self.assertEqual(tuple(output.shape), tuple(x.shape))
        torch.testing.assert_close(output, x)

        # prefix=2, block-size=4: token 2 closes page 3; token 5 ends page 4.
        torch.testing.assert_close(
            conv_states[3],
            torch.tensor([[102, 10, 20], [202, 11, 21]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            conv_states[4],
            torch.tensor([[30, 40, 50], [31, 41, 51]], dtype=torch.float32),
        )

    def test_decode_is_tokenwise_and_copies_state_when_page_changes(self):
        # conv_state uses the AscendC pool layout (pages, state_len, dim)
        x = torch.tensor([[[20, 21], [30, 31]]], dtype=torch.float16)
        weight = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        conv_state = torch.zeros(6, 3, 2, dtype=torch.float32)
        conv_state[2] = torch.tensor(
            [[1, 11], [2, 12], [3, 13]], dtype=torch.float32
        )
        block_map = torch.tensor([[2, 5]], dtype=torch.int32)
        sequence_lengths = torch.tensor([2], dtype=torch.int32)
        calls = []

        def fake_npu_causal_conv1d(**kwargs):
            calls.append(
                {
                    "x_shape": tuple(kwargs["x"].shape),
                    "weight_shape": tuple(kwargs["weight"].shape),
                    "state_shape": tuple(kwargs["conv_states"].shape),
                    "cache_indices": list(kwargs["cache_indices"]),
                    "run_mode": kwargs["run_mode"],
                    "activation_mode": kwargs["activation_mode"],
                }
            )
            states = kwargs["conv_states"]
            for batch_index, page_index in enumerate(kwargs["cache_indices"]):
                if page_index == kwargs["pad_slot_id"]:
                    continue
                previous = states[page_index].clone()
                states[page_index, :-1].copy_(previous[1:])
                states[page_index, -1].copy_(kwargs["x"][batch_index])
            return kwargs["x"] + 100

        with mock.patch.object(
            self.module,
            "_load_npu_causal_conv1d",
            return_value=fake_npu_causal_conv1d,
        ):
            output = self.module.causal_conv1d_update(
                x=x,
                conv_state=conv_state,
                weight=weight,
                activation=True,
                block_map=block_map,
                seq_size_per_block=4,
                sequence_lengths=sequence_lengths,
            )

        self.assertEqual([call["cache_indices"] for call in calls], [[2], [5]])
        self.assertTrue(all(call["x_shape"] == (1, 2) for call in calls))
        self.assertTrue(all(call["weight_shape"] == (4, 2) for call in calls))
        self.assertTrue(all(call["state_shape"] == (6, 3, 2) for call in calls))
        self.assertTrue(all(call["run_mode"] == 1 for call in calls))
        self.assertTrue(all(call["activation_mode"] == 1 for call in calls))
        self.assertEqual(tuple(output.shape), tuple(x.shape))
        self.assertEqual(output.dtype, x.dtype)
        torch.testing.assert_close(output, x + 100)

        # RTP snapshots every speculative token in its own consecutive block
        # entry.  Thus token 2 uses page 5 even though logical length 3 has not
        # crossed a normal block-size-4 boundary, and inherits updated page 2.
        torch.testing.assert_close(
            conv_state[5],
            torch.tensor([[3, 13], [20, 30], [21, 31]], dtype=torch.float32),
        )


if __name__ == "__main__":
    unittest.main()
