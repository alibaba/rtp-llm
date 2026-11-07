"""dsv41 must thread the real ParallelismConfig into the fused-MoE layer.

The dsv41 chain builds its ChunkedFp8Fp4MoeLayer from ``_block_kwargs``;
without the real config the layer falls back to a tp_size=1 namespace and
TP-sharded weights silently skip their Group.TP reduction.
"""

import unittest

from rtp_llm.models_py.modules.dsv41.transformer import V4Args, _block_kwargs


class Dsv41BlockKwargsTest(unittest.TestCase):
    def test_block_kwargs_threads_parallelism_config(self):
        config = object()
        kwargs = _block_kwargs(0, V4Args(parallelism_config=config), None)
        self.assertIs(kwargs.get("parallelism_config"), config)

    def test_block_kwargs_defaults_to_none(self):
        kwargs = _block_kwargs(0, V4Args(), None)
        self.assertIsNone(kwargs.get("parallelism_config"))


if __name__ == "__main__":
    unittest.main()
