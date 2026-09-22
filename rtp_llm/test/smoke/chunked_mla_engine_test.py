"""Real engine differential test with a small, locally generated MLA or hybrid checkpoint."""

import json
import math
import os
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import requests
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import StatusVersionPB
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub
from rtp_llm.test.utils.maga_server_manager import MagaServerManager


def make_checkpoint(path: Path, *, hybrid: bool = False):
    hidden, intermediate, heads, layers, vocab = 256, 512, 16, 2, 256
    config = {
        "architectures": ["DeepseekV2ForCausalLM"],
        "model_type": "deepseek_v2",
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_attention_heads": heads,
        "num_key_value_heads": heads,
        "num_hidden_layers": layers,
        "vocab_size": vocab,
        "max_position_embeddings": 4096,
        "q_lora_rank": None,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "first_k_dense_replace": layers,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": intermediate,
        "routed_scaling_factor": 1.0,
        "torch_dtype": "bfloat16",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_word_embeddings": False,
    }
    if hybrid:
        config.update(
            {
                "architectures": ["Qwen3NextForCausalLM"],
                "model_type": "qwen3_next",
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "head_dim": 64,
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "full_attention_interval": 2,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 2,
                "num_experts": 2,
                "shared_expert_intermediate_size": 0,
                "rms_norm_eps": 1e-6,
                "mamba_ssm_dtype": "float32",
            }
        )
    (path / "config.json").write_text(json.dumps(config))
    generator = torch.Generator(device="cpu").manual_seed(42)

    def weight(*shape):
        return (torch.randn(shape, generator=generator, device="cpu") * 0.02).bfloat16()

    # Qwen3Next's loader adds one to these RMSNorm weights.
    norm = torch.zeros if hybrid else torch.ones
    weights = {
        "model.embed_tokens.weight": weight(vocab, hidden),
        "model.norm.weight": norm(hidden, dtype=torch.bfloat16, device="cpu"),
        "lm_head.weight": weight(vocab, hidden),
    }
    mlp_shapes = {
        "gate_proj": (intermediate, hidden),
        "up_proj": (intermediate, hidden),
        "down_proj": (hidden, intermediate),
    }
    for layer in range(layers):
        prefix = f"model.layers.{layer}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            weights[prefix + name + ".weight"] = norm(
                hidden, dtype=torch.bfloat16, device="cpu"
            )
        if hybrid:
            shapes = {"mlp.gate": (2, hidden)}
            for expert in range(2):
                for name, shape in mlp_shapes.items():
                    shapes[f"mlp.experts.{expert}.{name}"] = shape
            if layer == 0:
                shapes.update(
                    {
                        "linear_attn.in_proj_qkvz": (1024, hidden),
                        "linear_attn.in_proj_ba": (4, hidden),
                        "linear_attn.conv1d": (768, 1, 4),
                        "linear_attn.out_proj": (hidden, 256),
                    }
                )
                weights[prefix + "linear_attn.norm.weight"] = torch.ones(
                    128, dtype=torch.bfloat16
                )
                weights[prefix + "linear_attn.A_log"] = torch.zeros(
                    2, dtype=torch.float32
                )
                weights[prefix + "linear_attn.dt_bias"] = torch.zeros(
                    2, dtype=torch.bfloat16
                )
            else:
                shapes.update(
                    {
                        "self_attn.q_proj": (512, hidden),
                        "self_attn.k_proj": (256, hidden),
                        "self_attn.v_proj": (256, hidden),
                        "self_attn.o_proj": (hidden, 256),
                    }
                )
                for name in ("q_norm", "k_norm"):
                    weights[prefix + "self_attn." + name + ".weight"] = norm(
                        64, dtype=torch.bfloat16
                    )
        else:
            shapes = {
                "self_attn.q_proj": (heads * 192, hidden),
                "self_attn.kv_a_proj_with_mqa": (576, hidden),
                "self_attn.kv_b_proj": (heads * 256, 512),
                "self_attn.o_proj": (hidden, heads * 128),
            }
            shapes.update({f"mlp.{name}": shape for name, shape in mlp_shapes.items()})
            weights[prefix + "self_attn.kv_a_layernorm.weight"] = torch.ones(
                512, dtype=torch.bfloat16, device="cpu"
            )
        for name, shape in shapes.items():
            weights[prefix + name + ".weight"] = weight(*shape)
    save_file(weights, str(path / "model.safetensors"))
    vocabulary = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2}
    vocabulary.update({f"t{i}": i for i in range(3, vocab)})
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    fast.chat_template = (
        "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    )
    fast.save_pretrained(path)


class ChunkedMlaEngineTest(unittest.TestCase):
    @staticmethod
    def _prompt(length, seed):
        return " ".join(f"t{3 + (seed + i) % 253}" for i in range(length))

    @staticmethod
    def _payload(prompt, new_tokens=6):
        return {
            "prompt": prompt,
            "yield_generator": True,
            "generate_config": {
                "max_new_tokens": new_tokens,
                "min_new_tokens": new_tokens,
                "top_k": 1,
                "return_output_ids": True,
                "reuse_cache": True,
                "is_streaming": True,
            },
        }

    @staticmethod
    def _frames(response):
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            text = line.decode().removeprefix("data:").strip()
            if text and text.lower() != "[done]":
                yield json.loads(text)

    def _request(self, manager, prompt, budget, reuse=0, concurrent=False):
        with requests.post(
            f"http://127.0.0.1:{manager.port}/",
            json=self._payload(prompt),
            stream=True,
            timeout=120,
        ) as response:
            frames = list(self._frames(response))
        # Intermediate chunks produce no output; then decode emits six tokens.
        self.assertEqual(len(frames), 6, frames)
        length = len(prompt.split())
        for step, frame in enumerate(frames, 1):
            info = frame["aux_info"]
            self.assertEqual(info["input_len"], length)
            self.assertEqual(info["output_len"], step)
            self.assertEqual(info["reuse_len"], reuse)
            iterations = (
                (math.ceil((length - reuse) / budget) if budget else 1) + step - 1
            )
            if concurrent:
                self.assertGreaterEqual(info["iter_count"], iterations)
            else:
                self.assertEqual(info["iter_count"], iterations)
        self.assertTrue(frames[-1]["finished"])
        return [frame["output_ids"] for frame in frames]

    def _wait_idle(self, stub, available=None):
        deadline = time.monotonic() + 10
        while True:
            status = stub.GetWorkerStatus(
                StatusVersionPB(latest_finished_version=-1), timeout=2
            )
            if not status.running_task_info and (
                available is None or status.available_kv_cache == available
            ):
                return status
            self.assertLess(
                time.monotonic(), deadline, "requests or KV slots not released"
            )
            time.sleep(0.02)

    def _cancel_and_retry(self, manager, prompt, budget, expected):
        # Cancel during decode: no model hooks or timing assumptions about prefill.
        # Cache reuse is disabled for this phase, so all request slots must return.
        with grpc.insecure_channel(f"127.0.0.1:{manager.port + 1}") as channel:
            stub = RpcServiceStub(channel)
            before = self._wait_idle(stub)
            self.assertGreater(before.available_kv_cache, 0)
            with requests.post(
                f"http://127.0.0.1:{manager.port}/",
                json=self._payload(prompt, new_tokens=1024),
                stream=True,
                timeout=120,
            ) as response:
                frames = self._frames(response)
                first = next(frames)
                self.assertFalse(first["finished"])
                self.assertEqual(first["aux_info"]["output_len"], 1)
                active = stub.GetWorkerStatus(
                    StatusVersionPB(latest_finished_version=-1), timeout=2
                )
                self.assertEqual(len(active.running_task_info), 1)
                request_id = active.running_task_info[0].request_id
                self.assertLess(active.available_kv_cache, before.available_kv_cache)
            after = self._wait_idle(stub, before.available_kv_cache)
            cancelled = [
                task
                for task in after.finished_task_list
                if task.request_id == request_id
            ]
            self.assertEqual(len(cancelled), 1)
            self.assertIn("cancelled", cancelled[0].error_info.error_message)
        self.assertEqual(self._request(manager, prompt, budget), expected)

    def test_full_vs_chunked_prefill_and_decode(self):
        model_type = os.environ.get("CHUNK_TEST_MODEL_TYPE", "deepseek2")
        tp = int(os.environ.get("MLA_TEST_TP_SIZE", "1"))
        block = int(os.environ.get("CHUNK_TEST_BLOCK_SIZE", "64"))
        kernel_block = int(os.environ.get("CHUNK_TEST_KERNEL_BLOCK_SIZE", str(block)))
        prompts = {
            "short": self._prompt(130, 0),
            "long": self._prompt(257, 47),
            "fork": self._prompt(128, 0) + " " + self._prompt(129, 149),
        }
        reuse_lengths = {
            name: length // block * block
            for name, length in (("short", 130), ("long", 257), ("fork", 128))
        }
        reference = {}
        with tempfile.TemporaryDirectory() as directory:
            make_checkpoint(Path(directory), hybrid=model_type == "qwen3_next")
            # One ordinary baseline, one chunk run, and one prefix-reuse run.
            for budget, reuse in ((0, False), (block, False), (block, True)):
                with self.subTest(budget=budget, reuse=reuse):
                    manager = MagaServerManager(
                        env_args={
                            "DETERMINISTIC_GEMM": "1",
                            "ENABLE_STABLE_SCATTER_ADD": "ON",
                        },
                        role_name=f"mla_chunk_{budget}_reuse_{int(reuse)}",
                        smoke_args_str=(
                            f"--act_type bf16 --tp_size {tp} --dp_size 1 --ep_size 1 --world_size {tp} "
                            f"--role_type PDFUSION --reuse_cache {int(reuse)} --prefill_chunk_size {budget} "
                            "--enable_device_cache 1 --enable_memory_cache 0 --enable_remote_cache 0 "
                            f"--enable_cuda_graph 0 --fp8_kv_cache 0 --seq_size_per_block {block} "
                            f"--kernel_seq_size_per_block {kernel_block} --test_block_num 128 --max_seq_len 2048 "
                            "--max_context_batch_size 2 --warm_up 0 --frontend_server_count 1 --shutdown_timeout 5"
                        ),
                    )
                    try:
                        self.assertTrue(
                            manager.start_server(
                                model_path=directory,
                                tokenizer_path=directory,
                                model_type=model_type,
                                timeout=300,
                            )
                        )
                        for name, prompt in prompts.items():
                            hit = reuse_lengths[name] if reuse and name == "fork" else 0
                            result = self._request(manager, prompt, budget, hit)
                            if budget == 0:
                                reference[name] = result
                            else:
                                self.assertEqual(result, reference[name])
                        # Repeat a short-tail prompt: the partial block must be recomputed.
                        if reuse:
                            self.assertEqual(
                                self._request(
                                    manager,
                                    prompts["short"],
                                    budget,
                                    reuse_lengths["short"],
                                ),
                                reference["short"],
                            )
                            with ThreadPoolExecutor(max_workers=2) as pool:
                                pending = {
                                    name: pool.submit(
                                        self._request,
                                        manager,
                                        prompts[name],
                                        budget,
                                        reuse_lengths[name],
                                        True,
                                    )
                                    for name in ("short", "long")
                                }
                                for name, future in pending.items():
                                    self.assertEqual(future.result(), reference[name])
                        if budget and not reuse:
                            self._cancel_and_retry(
                                manager, prompts["short"], budget, reference["short"]
                            )
                    finally:
                        manager.stop_server()


if __name__ == "__main__":
    unittest.main()
