"""Pure norm ownership must not eagerly import unrelated native collectives."""

import subprocess
import sys
import unittest


class LayerImportTest(unittest.TestCase):
    def test_norm_public_export_preserves_identity_without_native_import(self):
        script = """
import sys
import torch
import rtp_llm.models_py.layers as layers
from rtp_llm.models_py.layers import RMSNorm
from rtp_llm.models_py.layers.norm import RMSNorm as actual
assert RMSNorm is actual and layers.RMSNorm is actual
assert set(layers.__all__).issubset(dir(layers))
assert 'rtp_llm.models_py.layers.embedding' not in sys.modules
assert 'rtp_llm.models_py.distributed.collective_torch' not in sys.modules
assert 'rtp_llm.ops' not in sys.modules
module = RMSNorm(4, params_dtype=torch.bfloat16)
module.load_weights({'weight': torch.arange(1, 5).bfloat16()})
module.validate_weights_loaded()
try:
    layers.not_a_layer
except AttributeError:
    pass
else:
    raise AssertionError('unknown exports must remain errors')
"""
        subprocess.run([sys.executable, "-c", script], check=True)


if __name__ == "__main__":
    unittest.main()
