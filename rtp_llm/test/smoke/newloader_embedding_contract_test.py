"""Real NewLoader TP1/TP2 HTTP deployments versus an independent HF reference.

Dedicated GPU smoke: missing devices/checkpoints are failures, never skips.
Run a single model with ``... BertDeploymentSmoke`` or ``RobertaDeploymentSmoke``.
"""

import json
import os
import unittest
from pathlib import Path

import requests
import torch

from rtp_llm.test.utils.device_resource import get_gpu_ids
from rtp_llm.test.utils.maga_server_manager import MagaServerManager


class EmbeddingDeploymentContract:
    model_type: str
    default_checkpoint: str

    def test_short_positions_tp1_tp2_match_reference_http(self):
        self._check_deployments((1, 2))

    def test_short_positions_tp1_matches_reference_http(self):
        self._check_deployments((1,))

    def _check_deployments(self, tp_sizes):
        self.assertTrue(torch.cuda.is_available(), "GPU smoke requires a working GPU")
        self.assertGreaterEqual(torch.cuda.device_count(), max(tp_sizes))
        devices = list(get_gpu_ids())[: max(tp_sizes)]
        checkpoint = os.environ.get("CHECKPOINT_PATH", self.default_checkpoint)
        config = json.loads((Path(checkpoint) / "config.json").read_text())
        self.assertGreater(config["max_position_embeddings"], 128)
        outputs = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "."))
        outputs.mkdir(parents=True, exist_ok=True)
        inputs = [
            "hello",
            "A smoke test compares real embedding inference.",
            "你好，世界",
        ]
        reference_mode = os.environ.get("NEWLOADER_SMOKE_REFERENCE", "hf")
        self.assertIn(reference_mode, ("hf", "legacy"))
        reference = None
        if reference_mode == "hf":
            from sentence_transformers import SentenceTransformer

            from rtp_llm.frontend.tokenizer_factory.tokenizers.bert_tokenizer import (
                BertTokenizer,
            )

            # CPU reference uses the checkpoint's own pooling/normalization
            # modules, independently of RTP-LLM's loaded weights and forward.
            # Feed identical service token IDs: bare HF v5 tokenization can
            # differ from BaseTokenizer's checkpoint-normalizer compatibility
            # handling (e.g. XLM-R full-width punctuation).
            reference_model = SentenceTransformer(
                checkpoint, device="cpu", local_files_only=True
            )
            service_tokenizer = BertTokenizer(checkpoint).tokenizer
            features = service_tokenizer(
                inputs,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            reference_model.eval()
            with torch.inference_mode():
                reference = reference_model(features)["sentence_embedding"].float()
            del reference_model
            (outputs / f"{self.model_type}_hf_reference.json").write_text(
                json.dumps(reference.tolist())
            )
            (outputs / f"{self.model_type}_reference_tokens.json").write_text(
                json.dumps(service_tokenizer(inputs)["input_ids"])
            )
        results = {}
        session = requests.Session()
        session.trust_env = False
        self.addCleanup(session.close)
        deployments = [(True, tp) for tp in tp_sizes]
        if reference_mode == "legacy":
            deployments.insert(0, (False, 1))
        for newloader, tp in deployments:
            role = f"{self.model_type}_{'newloader' if newloader else 'legacy'}_tp{tp}"
            manager = MagaServerManager(
                env_args={
                    "WORLD_SIZE": str(tp),
                    "USE_NEW_LOADER": str(int(newloader)),
                    "REQUIRE_WEIGHT_UPDATE": "0",
                    "LOAD_METHOD": "scratch",
                    "LOG_LEVEL": "INFO",
                    "FT_SERVER_TEST": "1",
                },
                device_ids=devices[:tp],
                role_name=role,
                smoke_args_str=(
                    f"--tp_size {tp} --ep_size {tp} --world_size {tp} --dp_size 1 "
                    f"--use_new_loader {str(newloader).lower()} "
                    "--require_weight_update false --load_method scratch "
                    "--max_seq_len 128 --act_type FP16 --task_type DENSE_EMBEDDING "
                    "--seq_size_per_block 16 --enable_cuda_graph false --warm_up false "
                    "--concurrency_limit 2 --max_context_batch_size 2 --kv_cache_mem_mb 256 "
                    "--use_local true --use_local_preprocess true --frontend_server_count 1 "
                    "--shutdown_timeout 10 --frontend_pre_stop_drain_seconds 0 "
                    "--dash_sc_grpc_pre_stop_drain_seconds 0 "
                    + ("--use_aiter_pa 1 --use_asm_pa 1 " if torch.version.hip else "")
                ),
            )
            self.addCleanup(manager.stop_server)
            try:
                self.assertTrue(
                    manager.start_server(checkpoint, self.model_type, timeout=900),
                    f"{role} failed; see {manager.log_file_path}",
                )
                response = session.post(
                    f"http://127.0.0.1:{manager.port}/v1/embeddings",
                    json={"input": inputs, "model": "smoke"},
                    # A cold ROCm worker may compile AITER on its first
                    # embedding request, after HTTP readiness has succeeded.
                    timeout=900,
                )
                response.raise_for_status()
                payload = response.json()
                (outputs / f"{role}.json").write_text(json.dumps(payload))
                rows = sorted(payload["data"], key=lambda row: row["index"])
                self.assertEqual(
                    [row["index"] for row in rows], list(range(len(inputs)))
                )
                result = torch.tensor([row["embedding"] for row in rows])
                self.assertEqual(
                    tuple(result.shape), (len(inputs), config["hidden_size"])
                )
                self.assertTrue(torch.isfinite(result).all().item())
                self.assertTrue((result.norm(dim=-1) > 0).all().item())
                results[newloader, tp] = result
                log = Path(manager.log_file_path).read_text(errors="replace")
                route = "newloader" if newloader else "legacy"
                self.assertIn(
                    f"loader route for model_type={self.model_type}: {route}", log
                )
                if not newloader:
                    reference = result
                else:
                    self.assertIsNotNone(reference)
                    torch.testing.assert_close(result, reference, rtol=1e-2, atol=1e-2)
                    cosine = torch.nn.functional.cosine_similarity(result, reference)
                    self.assertTrue((cosine > 0.999).all().item(), cosine.tolist())
                    (outputs / f"{role}_verified.json").write_text(
                        json.dumps(
                            {
                                "reference": reference_mode,
                                "min_cosine": cosine.min().item(),
                                "max_abs_error": (result - reference)
                                .abs()
                                .max()
                                .item(),
                                "passed": True,
                            }
                        )
                    )
            finally:
                manager.stop_server()
        if 2 in tp_sizes:
            torch.testing.assert_close(
                results[True, 2], results[True, 1], rtol=1e-2, atol=1e-2
            )
        topology = "_".join(f"tp{tp}" for tp in tp_sizes)
        (outputs / f"{self.model_type}_{topology}_contract_passed.json").write_text(
            json.dumps(
                {
                    "checkpoint": checkpoint,
                    "reference": reference_mode,
                    "newloader_tp": list(tp_sizes),
                    "max_seq_len": 128,
                    "requests": len(inputs),
                    "passed": True,
                }
            )
        )


class BertDeploymentSmoke(EmbeddingDeploymentContract, unittest.TestCase):
    model_type = "bert"
    default_checkpoint = "/mnt/nas1/hf/bge-large-en-v1.5"


class RobertaDeploymentSmoke(EmbeddingDeploymentContract, unittest.TestCase):
    model_type = "roberta"
    default_checkpoint = "/mnt/nas1/hf/bge-m3"


if __name__ == "__main__":
    unittest.main()
