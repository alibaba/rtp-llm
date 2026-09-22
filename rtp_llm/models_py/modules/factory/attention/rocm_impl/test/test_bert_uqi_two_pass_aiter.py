import os
from pathlib import Path
import json
import sys
import importlib.metadata
import unittest
import torch
from bert_uqi_test_utils import check_cases


class TestBertUqiAiter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A selected MI308 test without ROCm is an infrastructure FAILURE, never skip.
        if torch.version.hip is None or not torch.cuda.is_available():
            raise RuntimeError("This target requires a real ROCm GPU and AITER")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one prechecked GPU")
        class NoFlashinfer:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "flashinfer" or fullname.startswith("flashinfer."):
                    raise AssertionError("ROCm operator test attempted to import FlashInfer")
        sys.meta_path.insert(0, NoFlashinfer())
        from aiter import flash_attn_varlen_func
        print("RUNTIME", json.dumps(dict(python=sys.executable, torch=torch.__version__,
            hip=torch.version.hip, gpu=torch.cuda.get_device_name(0),
            aiter=importlib.metadata.version("aiter"), triton=importlib.metadata.version("triton"))), flush=True)
        import aiter.jit.core as jit_core
        paths = dict(aiter_root=jit_core.AITER_ROOT_DIR,
                     aiter_meta=jit_core.AITER_META_DIR,
                     aiter_jit=os.environ["AITER_JIT_DIR"],
                     triton_cache=os.environ["TRITON_CACHE_DIR"])
        for name, value in paths.items():
            path = Path(value)
            if not path.is_absolute() or not os.access(path, os.W_OK):
                raise RuntimeError(f"{name} must be a writable absolute directory: {value}")
        print("CACHE_PATHS", json.dumps(paths), flush=True)
        cls.attention = staticmethod(flash_attn_varlen_func)

    def test_fp16(self):
        check_cases(self, "cuda", torch.float16, self.attention)

    def test_bf16(self):
        check_cases(self, "cuda", torch.bfloat16, self.attention)


if __name__ == "__main__":
    unittest.main()
