import unittest
from types import SimpleNamespace

from rtp_llm.models_py.mla_routes import resolve_mla_routes, run_sparse_indexer


class KimiK31RoutesTest(unittest.TestCase):
    def test_k3_full_and_indexer_routes_keep_their_own_cache(self):
        attention, indexer, full_cache, index_cache = (object() for _ in range(4))
        result = resolve_mla_routes(
            {"full": attention, "indexer_kv": indexer},
            {"full": full_cache, "indexer_kv": index_cache},
            sparse=True, attention_tag="full",
        )
        self.assertEqual(result, (attention, full_cache, indexer, index_cache))

    def test_dense_k3_route_stays_untagged(self):
        attention, cache = object(), object()
        self.assertEqual(resolve_mla_routes(attention, cache, sparse=False),
                         (attention, cache, None, None))
        with self.assertRaises(RuntimeError):
            resolve_mla_routes({"full": attention}, cache, sparse=False)

    def test_missing_or_extra_sparse_routes_are_rejected(self):
        for routes in ({"full": 1}, {"full": 1, "indexer_kv": 2, "linear": 3}):
            with self.subTest(routes=routes), self.assertRaises(RuntimeError):
                resolve_mla_routes(routes, None, sparse=True, attention_tag="full")
        with self.assertRaises(RuntimeError):
            resolve_mla_routes({"full": 1, "indexer_kv": 2}, {"full": 3},
                               sparse=True, attention_tag="full")

    def test_indexer_cache_is_published_after_its_write(self):
        events = []
        cache, hidden, latent, topk = (object() for _ in range(4))
        params, cp = object(), object()
        inputs = SimpleNamespace(
            is_prefill=True, cache_store_inputs=object(),
            cache_store_writer=SimpleNamespace(write=lambda request, value:
                                               events.append(("publish", request, value))),
        )
        fmha = SimpleNamespace(fmha_params=params, cp_params=cp, attn_inputs=inputs,
                               is_sparse=lambda: True)

        def indexer(x, q, value, actual_params, actual_inputs, **kwargs):
            self.assertIs(x, hidden)
            self.assertIs(q, latent)
            self.assertIs(value, cache)
            self.assertIs(actual_params, params)
            self.assertIs(actual_inputs, inputs)
            self.assertEqual(kwargs, {"use_fast_path": False, "cp_params": cp})
            events.append(("write", cache))
            return topk

        self.assertIs(run_sparse_indexer(indexer, hidden, latent, cache, fmha), topk)
        self.assertEqual(events, [("write", cache),
                                  ("publish", inputs.cache_store_inputs, cache)])


if __name__ == "__main__":
    unittest.main()
