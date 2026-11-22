import os
import types
import unittest
from unittest.mock import patch

from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.decode import writer_backend


class FlashMLAWriterBackendTest(unittest.TestCase):
    @staticmethod
    def _tensor(last_dim):
        return types.SimpleNamespace(size=lambda _index: last_dim)

    def test_proxy_dispatches_only_ppu_decode_shape(self):
        calls = []

        def stock_call(*args):
            calls.append(("stock", args))
            return ("stock-out", "stock-lse")

        def writer_call(*args):
            calls.append(("writer", args))
            return ("writer-out", "writer-lse", "metadata", "splits")

        stock = types.SimpleNamespace(fwd_kvcache_mla=stock_call)
        writer = types.SimpleNamespace(sparse_decode_fwd=writer_call)
        proxy = writer_backend.DecodeNativeProxy(stock, writer)
        decode_args = (
            FlashMLAWriterBackendTest._tensor(512),  # q
            FlashMLAWriterBackendTest._tensor(584),  # k_cache
            object(),  # v_cache
            512,
            object(),  # cache_seqlens
            object(),  # block_table
            0.04,
            False,
            object(),  # metadata
            object(),  # splits
            True,
            object(),  # indices
            object(),  # sink
            object(),  # lengths
            object(),  # extra_kv
            object(),  # extra_indices
            object(),  # extra_lengths
            object(),  # out
        )
        self.assertEqual(proxy.fwd_kvcache_mla(*decode_args), ("writer-out", "writer-lse"))
        self.assertEqual(calls[-1][0], "writer")

        non_decode_args = list(decode_args)
        non_decode_args[10] = False
        proxy.fwd_kvcache_mla(*non_decode_args)
        self.assertEqual(calls[-1][0], "stock")

    def test_private_interface_does_not_mutate_stock(self):
        stock = types.SimpleNamespace(fwd_kvcache_mla=lambda *_: "stock")
        writer = types.SimpleNamespace(sparse_decode_fwd=lambda *_: "writer")
        namespace = {"flash_mla_cuda": stock}
        exec("def api(*args):\n    return flash_mla_cuda.fwd_kvcache_mla(*args)", namespace)
        original = namespace["api"]
        native = stock.fwd_kvcache_mla
        private = writer_backend.private_interface(original, stock, writer)
        self.assertIsNot(private, original)
        self.assertIs(stock.fwd_kvcache_mla, native)

    def test_writer_backend_is_role_gated(self):
        with patch.object(writer_backend, "load_writer_backend", return_value="writer") as load:
            cases = (
                ("PREFILL", "/writer.so", None),
                ("DECODE", "", None),
                ("DECODE", "/writer.so", "writer"),
            )
            for role, path, expected in cases:
                with patch.dict(os.environ, {"ROLE_TYPE": role, "DSV4_PPU_FLASHMLA_WRITER_PATH": path}):
                    self.assertEqual(writer_backend.get_writer_backend(), expected)
            self.assertEqual(load.call_count, 1)


if __name__ == "__main__":
    unittest.main()
