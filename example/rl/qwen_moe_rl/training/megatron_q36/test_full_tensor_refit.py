import asyncio
import contextlib
import io
import sys
import types
import unittest
from unittest.mock import patch

import full_tensor_refit


class Tensor:
    def __init__(self, dtype, value):
        self.dtype, self.value = dtype, value

    def float(self):
        return Tensor("torch.float32", self.value)

    def detach(self):
        return self


class FullTensorRefitTests(unittest.TestCase):
    def test_driver_hook_chains_existing_hook_and_rejects_others(self):
        env = types.SimpleNamespace(get_ppo_ray_runtime_env=lambda: {
            "worker_process_setup_hook": "install_rtp_bridge.worker_setup",
            "env_vars": {"EXAMPLE": "preserved"},
        })
        full_tensor_refit.install_driver_hook(env)
        result = env.get_ppo_ray_runtime_env()
        self.assertEqual(result["worker_process_setup_hook"], "full_tensor_refit.worker_setup")
        self.assertEqual(result["env_vars"]["EXAMPLE"], "preserved")
        existing = env.get_ppo_ray_runtime_env
        full_tensor_refit.install_driver_hook(env)
        self.assertIs(env.get_ppo_ray_runtime_env, existing)
        other = types.SimpleNamespace(get_ppo_ray_runtime_env=lambda: {"worker_process_setup_hook": "other.hook"})
        full_tensor_refit.install_driver_hook(other)
        with self.assertRaisesRegex(RuntimeError, "different Ray worker"):
            other.get_ppo_ray_runtime_env()

    def test_delegation_upcast_order_and_arguments(self):
        class FakeAdapter:
            async def update_weights(self, weights, global_steps=None, **kwargs):
                # Match the existing RTP adapter: access dtype and detach inside
                # its per-tensor error-handling block, after name coordination.
                self.observed = [(name, value.dtype, value.detach()) for name, value in weights]
                self.arguments = global_steps, kwargs
                return "underlying-result"

        full_tensor_refit.install(FakeAdapter)
        tensor = Tensor("torch.bfloat16", 7)
        raw = [("model.language_model.layers.0.linear_attn.A_log", tensor),
               ("model.language_model.layers.0.mlp.gate.weight", tensor)]
        adapter = FakeAdapter()
        fake_torch = types.SimpleNamespace(float32="torch.float32")
        with patch.dict(sys.modules, {"torch": fake_torch}), contextlib.redirect_stdout(io.StringIO()) as output:
            result = asyncio.run(adapter.update_weights(iter(raw), global_steps=4, peft_config=None))
        self.assertEqual(result, "underlying-result")
        self.assertEqual(adapter.arguments, (4, {"peft_config": None}))
        self.assertEqual(adapter.observed[0][1], "torch.float32")
        self.assertEqual(adapter.observed[0][2].dtype, "torch.float32")
        self.assertEqual(adapter.observed[0][2].value, 7)
        self.assertIs(adapter.observed[1][2], tensor)
        self.assertIn('"gdn_bf16_to_fp32": 1', output.getvalue())

    def test_original_error_propagates_and_no_success_log(self):
        class FakeAdapter:
            async def update_weights(self, weights, global_steps=None, **kwargs):
                list(weights)
                raise RuntimeError("native ACK failed")

        full_tensor_refit.install(FakeAdapter)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaisesRegex(RuntimeError, "native ACK failed"):
                asyncio.run(FakeAdapter().update_weights([], global_steps=2))
        self.assertEqual(output.getvalue(), "")

    def test_install_is_idempotent(self):
        class FakeAdapter:
            async def update_weights(self, weights, global_steps=None, **kwargs):
                return None

        first = full_tensor_refit.install(FakeAdapter)
        self.assertIs(full_tensor_refit.install(FakeAdapter), first)


if __name__ == "__main__":
    unittest.main()
