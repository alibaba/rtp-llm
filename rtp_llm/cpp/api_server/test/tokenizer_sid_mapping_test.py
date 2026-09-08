import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

from tokenizer_sid_mapping_test_lib import sid_mapping_json


# Load the actual BaseTokenizer without importing the unrelated tokenizer registry.
_source = Path(__file__).resolve().parents[3] / "frontend/tokenizer_factory/tokenizers/base_tokenizer.py"
_spec = importlib.util.spec_from_file_location("sid_test_base_tokenizer", _source)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
BaseTokenizer = _module.BaseTokenizer


class Vocabulary:
    eos_token_id = 1999
    # HF vocab_size may exclude added tokens; use __len__, not this property.
    vocab_size = 1800

    def __init__(self, vocab=None):
        self.vocab = vocab if vocab is not None else {"C2": 1902, "C10": 1910, "word": 7, "Cbad": 8}

    def get_vocab(self):
        return self.vocab

    def __len__(self):
        return 2000


def wrapped(vocab, eos=None):
    wrapper = BaseTokenizer.__new__(BaseTokenizer)
    wrapper.tokenizer = vocab
    wrapper.config_json = {} if eos is None else {"eos_token_id": eos}
    return wrapper


class TokenizerSidMappingTest(unittest.TestCase):
    def manifest(self, tokenizer):
        return json.loads(sid_mapping_json(tokenizer))

    def test_direct_tokenizer_and_actual_base_wrapper_match(self):
        raw = Vocabulary()
        wrapper = wrapped(raw)
        self.assertFalse(hasattr(wrapper, "get_vocab"))
        self.assertFalse(hasattr(wrapper, "__len__"))
        direct = self.manifest(raw)
        self.assertEqual(self.manifest(wrapper), direct)
        self.assertEqual(direct["tokens"], {"C2": 1902, "C10": 1910})
        self.assertEqual(direct["vocab_size"], 2000)
        self.assertEqual(direct["start_token_id"], 1699)
        self.assertEqual(direct["end_token_id"], 1999)
        canonical = "rtp-sid-mapping-v1\n2000\n1699\n1999\nC10\t1910\nC2\t1902\n"
        self.assertEqual(direct["mapping_fingerprint"], hashlib.sha256(canonical.encode()).hexdigest())

    def test_wrapper_effective_eos_fallback(self):
        raw = Vocabulary()
        raw.eos_token_id = None
        manifest = self.manifest(wrapped(raw, eos=1998))
        self.assertEqual(manifest["end_token_id"], 1998)
        self.assertNotEqual(manifest["mapping_fingerprint"], self.manifest(Vocabulary())["mapping_fingerprint"])

    def test_wrapper_override_not_underlying_eos(self):
        class Override(BaseTokenizer):
            @property
            def eos_token_id(self):
                return 1997

        wrapper = Override.__new__(Override)
        wrapper.tokenizer = Vocabulary()
        self.assertEqual(self.manifest(wrapper)["end_token_id"], 1997)

    def test_fingerprint_independent_of_vocabulary_insertion_order(self):
        self.assertEqual(self.manifest(Vocabulary({"C2": 1902, "C10": 1910})),
                         self.manifest(wrapped(Vocabulary({"C10": 1910, "C2": 1902}))))

    def test_no_c_tokens_has_no_manifest(self):
        self.assertEqual(self.manifest(wrapped(Vocabulary({"hello": 3}))), {})

    def test_unsupported_underlying_tokenizer_has_no_manifest(self):
        for raw in (None, object()):
            with self.subTest(raw=raw):
                self.assertEqual(self.manifest(wrapped(raw)), {})

    def test_missing_length_has_no_manifest(self):
        class NoLength:
            def get_vocab(self):
                return {"C1": 1900}

        self.assertEqual(self.manifest(wrapped(NoLength())), {})

    def test_invalid_eos_has_no_manifest(self):
        for eos in (None, -1, 2000):
            with self.subTest(eos=eos):
                raw = Vocabulary()
                raw.eos_token_id = eos
                self.assertEqual(self.manifest(raw), {})

    def test_invalid_c_token_fails_closed(self):
        for token_id in (-1, 2000, 1699, 1999):
            with self.subTest(token_id=token_id):
                with self.assertRaisesRegex(RuntimeError, "invalid C-token"):
                    sid_mapping_json(wrapped(Vocabulary({"C0": token_id})))

    def test_broken_unwrapper_does_not_fall_back_to_other_vocabulary(self):
        class Broken(Vocabulary):
            def get_real_tokenizer(self):
                raise ValueError("unwrapper failed")

        with self.assertRaisesRegex(ValueError, "unwrapper failed"):
            sid_mapping_json(Broken())


if __name__ == "__main__":
    unittest.main()
