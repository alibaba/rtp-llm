import subprocess
import sys
import unittest


class CoreBoundaryTest(unittest.TestCase):
    def test_first_import_has_no_concrete_model_or_native_backend(self):
        # A fresh process protects the actual import boundary, not a preloaded
        # module with selected cache fields cleared after the fact.
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; "
                "from rtp_llm.models_py.model_loader import NewModelLoader; "
                "from rtp_llm.models_py.registry import list_models; "
                "assert list_models() == [], list_models(); "
                "forbidden = ('rtp_llm.models_py.new_models', "
                "'rtp_llm.ops', 'libth_transformer_config', "
                "'librtp_compute_ops', 'deep_gemm', 'flashinfer'); "
                "loaded = [name for name in sys.modules "
                "if any(name == item or name.startswith(item + '.') "
                "for item in forbidden)]; "
                "assert not loaded, loaded",
            ],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
