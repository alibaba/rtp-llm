"""dsv41 must thread the real topology and MoE config into the fused-MoE layer.

The dsv41 chain builds its ChunkedFp8Fp4MoeLayer from ``_block_kwargs``;
without the real ParallelismConfig the layer falls back to a tp_size=1
namespace and TP-sharded weights silently skip their Group.TP reduction,
and without the MoEConfig/engine graph flag the DeepEP selection and
graph-safety gates run on invented per-layer fallbacks.
"""

import unittest

from rtp_llm.models_py.modules.dsv41.transformer import V4Args, _block_kwargs


class Dsv41BlockKwargsTest(unittest.TestCase):
    def test_block_kwargs_threads_parallelism_config(self):
        config = object()
        kwargs = _block_kwargs(0, V4Args(parallelism_config=config), None)
        self.assertIs(kwargs.get("parallelism_config"), config)

    def test_block_kwargs_threads_moe_config_and_graph_flag(self):
        moe_config = object()
        kwargs = _block_kwargs(
            0, V4Args(moe_config=moe_config, enable_cuda_graph=True), None
        )
        self.assertIs(kwargs.get("moe_config"), moe_config)
        self.assertTrue(kwargs.get("enable_cuda_graph"))

    def test_block_kwargs_defaults(self):
        kwargs = _block_kwargs(0, V4Args(), None)
        self.assertIsNone(kwargs.get("parallelism_config"))
        self.assertIsNone(kwargs.get("moe_config"))
        self.assertFalse(kwargs.get("enable_cuda_graph"))


if __name__ == "__main__":
    unittest.main()
