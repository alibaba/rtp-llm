import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.device import get_current_device
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.base_model import BaseModel
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.model_weight import CkptWeightInfo, W


class LoadEmbeddingWeightTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.expected = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        self.ft_key = ModelWeights.global_weight_prefix(2, 3, 4) + W.embedding
        save_file(
            {
                "model.embed_tokens.weight": self.expected,
                self.ft_key: self.expected + 1,
            },
            str(Path(self.directory.name) / "model.safetensors"),
        )
        self.loader = ModelLoader.__new__(ModelLoader)
        self.loader._load_config = SimpleNamespace(
            database=CkptDatabase(self.directory.name),
            is_ft_style_weight=False,
            compute_dtype=torch.float16,
            tp_rank=2,
            dp_rank=3,
            ep_rank=4,
            tp_size=1,
            dp_size=1,
            ep_size=1,
            merge_lora=False,
            exported_device=get_current_device(),
        )
        self.loader._model_weights_info = SimpleNamespace(
            weights=[
                AtomicWeight(W.embedding, [CkptWeightInfo("model.embed_tokens.weight")])
            ]
        )
        self.addCleanup(self.loader.cleanup_database)

    def test_real_weight_module_and_ft_key_load(self):
        for ft_style in (False, True):
            self.loader._load_config.is_ft_style_weight = ft_style
            with patch.object(
                self.loader,
                "force_clean_cuda_memory",
                wraps=self.loader.force_clean_cuda_memory,
            ) as clean:
                result = self.loader.load_embedding_weight("cpu")
            self.assertEqual(set(result), {W.embedding})
            torch.testing.assert_close(
                result[W.embedding], (self.expected + int(ft_style)).half()
            )
            self.assertEqual(result[W.embedding].device.type, "cpu")
            clean.assert_called_once_with()

    def test_missing_embedding_returns_none(self):
        self.loader._model_weights_info = None
        self.assertIsNone(self.loader.load_embedding_weight("cpu"))
        self.loader._model_weights_info = SimpleNamespace(weights=[])
        self.assertIsNone(self.loader.load_embedding_weight("cpu"))
        self.loader._load_config.is_ft_style_weight = True
        self.loader._load_config.tp_rank = 99
        self.assertIsNone(self.loader.load_embedding_weight("cpu"))

    def test_real_weight_failure_cleans_both_loader_boundaries(self):
        def fail_after_read(tensors):
            torch.testing.assert_close(tensors[0], self.expected.half())
            raise RuntimeError("embedding transform failed")

        self.loader._model_weights_info.weights[0].process_fun = fail_after_read
        model = BaseModel.__new__(BaseModel)
        with patch.object(
            model, "create_model_loader", return_value=self.loader
        ), patch.object(model, "_get_device_str", return_value="cpu"), patch.object(
            self.loader, "cleanup_database", wraps=self.loader.cleanup_database
        ) as database_clean, patch.object(
            self.loader,
            "force_clean_all_memory",
            wraps=self.loader.force_clean_all_memory,
        ) as all_clean, patch.object(
            self.loader,
            "force_clean_cuda_memory",
            wraps=self.loader.force_clean_cuda_memory,
        ) as cuda_clean:
            with self.assertRaisesRegex(RuntimeError, "embedding transform failed"):
                model.load_embedding_weight()
        cuda_clean.assert_called_once_with()
        database_clean.assert_called_once_with()
        all_clean.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
