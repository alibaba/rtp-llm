"""End-to-end Qwen2.5-Omni benchmark: thinker → talker → token2wav.

Three modes:
  --mode hf     : Use HuggingFace transformers for the full pipeline (reference)
  --mode hybrid : Use rtp-llm thinker + pure PyTorch talker/token2wav modules (legacy)
  --mode rtp    : Use rtp-llm thinker + rtp engine-based talker + token2wav

Usage:
    # HF reference (uses HF model.generate() for everything)
    CUDA_VISIBLE_DEVICES=5,6 python bench_omni_e2e.py \
        --ckpt /root/models/Qwen/Qwen2.5-Omni-7B \
        --mode hf --prompt "Tell me a joke." --output output_hf.wav

    # RTP engine mode: rtp-llm thinker + engine-based talker
    CUDA_VISIBLE_DEVICES=5 python bench_omni_e2e.py \
        --ckpt /root/models/Qwen/Qwen2.5-Omni-7B \
        --mode rtp --thinker-url http://localhost:18080 \
        --prompt "Tell me a joke." --output output_rtp.wav

    # Hybrid (legacy): rtp-llm thinker + pure PyTorch talker
    CUDA_VISIBLE_DEVICES=5 python bench_omni_e2e.py \
        --ckpt /root/models/Qwen/Qwen2.5-Omni-7B \
        --mode hybrid --thinker-url http://localhost:18080 \
        --prompt "Tell me a joke." --output output_hybrid.wav
"""
import argparse
import json
import logging
import os
import struct
import time
from typing import List, Optional, Tuple

import numpy as np
import requests
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("bench_omni_e2e")


def save_wav(waveform: torch.Tensor, path: str, sample_rate: int = 24000):
    audio = waveform.squeeze().detach().cpu().float().numpy()
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    with open(path, "wb") as f:
        num_samples = len(audio_int16)
        data_size = num_samples * 2
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))  # PCM
        f.write(struct.pack("<H", 1))  # mono
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * 2))
        f.write(struct.pack("<H", 2))
        f.write(struct.pack("<H", 16))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(audio_int16.tobytes())

    logger.info(f"Saved WAV: {path} ({num_samples/sample_rate:.2f}s, {sample_rate}Hz)")


# ============================================================
# Mode 1: Full HF transformers reference
# ============================================================

def run_hf_mode(args):
    """Run end-to-end using HuggingFace transformers Qwen2_5OmniForConditionalGeneration."""
    from transformers import AutoProcessor, AutoModelForCausalLM
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    t0 = time.time()

    logger.info("Loading HF model...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.ckpt,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    t_model = time.time()
    logger.info(f"Model loaded in {t_model - t0:.1f}s")

    processor = Qwen2_5OmniProcessor.from_pretrained(args.ckpt)
    t_proc = time.time()
    logger.info(f"Processor loaded in {t_proc - t_model:.1f}s")

    messages = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text_input, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    logger.info(f"Input token count: {inputs['input_ids'].shape[1]}")

    t_gen_start = time.time()
    text_ids, audio = model.generate(
        **inputs,
        speaker=args.speaker,
        thinker_max_new_tokens=args.max_thinker_tokens,
        talker_max_new_tokens=args.max_talker_tokens,
        use_audio_in_video=False,
    )
    t_gen_end = time.time()

    generated_text = processor.batch_decode(text_ids[:, inputs['input_ids'].shape[1]:],
                                            skip_special_tokens=True)[0]
    logger.info(f"Generated text: {generated_text}")
    logger.info(f"Audio shape: {audio.shape}")

    save_wav(audio, args.output)

    total = time.time() - t0
    logger.info("=== HF Summary ===")
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Text: {generated_text[:200]}")
    logger.info(f"Audio duration: {audio.numel() / 24000:.2f}s")
    logger.info(f"Generation time: {t_gen_end - t_gen_start:.2f}s")
    logger.info(f"Total time (incl. loading): {total:.2f}s")


# ============================================================
# Mode 2: Hybrid — rtp-llm thinker + our talker/token2wav
# ============================================================

def call_thinker_streaming(
    prompt: str,
    thinker_url: str,
    max_new_tokens: int = 256,
) -> Tuple[str, List[List[float]]]:
    """Call thinker engine with streaming to collect per-token hidden states."""
    logger.info(f"Calling thinker (streaming, max_tokens={max_new_tokens})...")

    resp = requests.post(
        thinker_url,
        json={
            "prompt": prompt,
            "generate_config": {
                "return_hidden_states": True,
                "max_new_tokens": max_new_tokens,
            },
            "stream": True,
        },
        stream=True,
    )
    resp.raise_for_status()

    all_hidden_states = []
    prev_text = ""

    for line in resp.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8")
        if line_str.startswith("data: "):
            line_str = line_str[6:]

        try:
            data = json.loads(line_str)
        except json.JSONDecodeError:
            continue

        hs = data.get("hidden_states")
        if hs is not None and len(hs) > 0:
            all_hidden_states.append(hs[0])

        text = data.get("response", "")
        if text:
            prev_text = text

    logger.info(f"Thinker generated: '{prev_text[:80]}...' ({len(all_hidden_states)} tokens)")
    return prev_text, all_hidden_states


def call_thinker_non_streaming_tokenize(prompt: str, thinker_url: str) -> List[int]:
    """Tokenize text using the thinker server."""
    try:
        resp = requests.post(f"{thinker_url}/tokenize", json={"prompt": prompt})
        if resp.status_code == 200:
            data = resp.json()
            return data.get("token_ids", data.get("tokens", []))
    except Exception:
        pass
    return []


def run_hybrid_mode(args):
    """Run e2e with rtp-llm thinker + our pure PyTorch talker/token2wav."""
    from rtp_llm.omni.models.qwen2_5_omni.talker_inference import TalkerInference
    from rtp_llm.omni.models.qwen2_5_omni.token2wav_model import Token2WavModel

    device = args.device
    t0 = time.time()

    # Load models
    logger.info("=== Loading models ===")
    t_load_start = time.time()
    talker = TalkerInference.from_pretrained(args.ckpt, device=device)
    t1 = time.time()
    logger.info(f"Talker loaded in {t1 - t_load_start:.1f}s")

    token2wav = Token2WavModel.from_pretrained(args.ckpt, device=device)
    t2 = time.time()
    logger.info(f"Token2Wav loaded in {t2 - t1:.1f}s")

    # Load thinker embedding layer
    from safetensors.torch import load_file
    index_path = os.path.join(args.ckpt, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    embed_key = "thinker.model.embed_tokens.weight"
    shard_file = index["weight_map"][embed_key]
    logger.info(f"Loading thinker embeddings from {shard_file}...")
    weights = load_file(os.path.join(args.ckpt, shard_file))
    embed_weight = weights[embed_key]
    thinker_embed = torch.nn.Embedding(embed_weight.shape[0], embed_weight.shape[1])
    thinker_embed.weight = torch.nn.Parameter(embed_weight)
    thinker_embed = thinker_embed.to(device=device, dtype=torch.bfloat16).eval()
    t3 = time.time()
    logger.info(f"Thinker embeddings loaded in {t3 - t2:.1f}s")

    # Load speaker data
    spk_dict = torch.load(os.path.join(args.ckpt, "spk_dict.pt"), map_location=device)
    speaker_data = spk_dict[args.speaker]
    speaker_bos = speaker_data["bos_token"]
    cond = speaker_data["cond"].float().to(device)
    ref_mel = speaker_data["ref_mel"].float().to(device)
    logger.info(f"Speaker: {args.speaker}, BOS: {speaker_bos}")

    # Run thinker
    logger.info("=== Running thinker ===")
    t_thinker_start = time.time()
    text, per_token_hs = call_thinker_streaming(args.prompt, args.thinker_url, args.max_thinker_tokens)
    t_thinker_end = time.time()
    logger.info(f"Thinker: {t_thinker_end - t_thinker_start:.2f}s, {len(per_token_hs)} tokens")
    logger.info(f"Text: {text[:200]}")

    if not per_token_hs:
        logger.error("No hidden states from thinker!")
        return

    # Tokenize generated text to get token IDs for embedding lookup
    gen_token_ids = call_thinker_non_streaming_tokenize(text, args.thinker_url)
    if not gen_token_ids:
        logger.error("Could not tokenize generated text")
        return

    # Also tokenize the prompt
    prompt_token_ids = call_thinker_non_streaming_tokenize(args.prompt, args.thinker_url)
    if not prompt_token_ids:
        logger.warning("Could not tokenize prompt, using fallback")
        prompt_token_ids = [0]

    logger.info(f"Prompt tokens: {len(prompt_token_ids)}, Generated tokens: {len(gen_token_ids)}")

    # Align counts
    num_gen = min(len(gen_token_ids), len(per_token_hs))
    gen_token_ids = gen_token_ids[:num_gen]
    per_token_hs = per_token_hs[:num_gen]

    dtype = torch.bfloat16

    # Build prompt hidden states and embeddings
    # For prompt: we don't have per-token hidden states from streaming,
    # so we use the thinker embedding as both hidden state and embedding.
    # This is an approximation — the HF approach uses the actual last-layer
    # hidden states for the prompt. For a first working version, we use
    # the embeddings only (which contain the token semantics).
    prompt_ids_t = torch.tensor(prompt_token_ids, dtype=torch.long, device=device)
    prompt_embeds = thinker_embed(prompt_ids_t).unsqueeze(0).to(dtype)  # [1, prompt_len, 3584]
    prompt_len = prompt_embeds.shape[1]

    # thinker_hidden_states[0] = prompt hidden (using embeddings as approximation)
    # thinker_token_embeds[0] = prompt embeddings (exact)
    thinker_hidden_states = [prompt_embeds]
    thinker_token_embeds = [prompt_embeds.clone()]

    # Add per-generated-token data
    gen_ids_t = torch.tensor(gen_token_ids, dtype=torch.long, device=device)
    gen_embeds = thinker_embed(gen_ids_t).to(dtype)  # [num_gen, 3584]

    for i in range(num_gen):
        hs_t = torch.tensor(per_token_hs[i], dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        embed_t = gen_embeds[i:i+1].unsqueeze(0)
        thinker_hidden_states.append(hs_t)
        thinker_token_embeds.append(embed_t)

    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)

    # Run talker
    logger.info("=== Running talker ===")
    t_talker_start = time.time()
    codec_tokens = talker.generate(
        thinker_hidden_states=thinker_hidden_states,
        thinker_token_embeds=thinker_token_embeds,
        input_ids=input_ids,
        speaker_bos_token=speaker_bos,
        thinker_embed_tokens=thinker_embed,
        max_new_tokens=args.max_talker_tokens,
        temperature=0.9,
        top_k=40,
        top_p=0.8,
        repetition_penalty=1.05,
    )
    t_talker_end = time.time()
    num_codec = codec_tokens.shape[1]
    logger.info(f"Talker: {t_talker_end - t_talker_start:.2f}s, {num_codec} codec tokens")
    logger.info(f"Codec tokens (first 30): {codec_tokens[0, :30].tolist()}")

    if num_codec == 0:
        logger.error("No codec tokens generated!")
        return

    # Filter special tokens
    mask = codec_tokens[0] < 8292
    codec_filtered = codec_tokens[0][mask].unsqueeze(0)
    if codec_filtered.shape[1] == 0:
        logger.error("All codec tokens were special. No audio.")
        return
    logger.info(f"Codec tokens after filter: {codec_filtered.shape[1]}")

    # Run token2wav
    logger.info("=== Running token2wav ===")
    t_t2w_start = time.time()
    with torch.no_grad():
        waveform = token2wav(codec_filtered.to(device), conditioning=cond, reference_mel=ref_mel)
    t_t2w_end = time.time()
    logger.info(f"Token2Wav: {t_t2w_end - t_t2w_start:.2f}s, waveform shape: {waveform.shape}")

    save_wav(waveform, args.output)

    total = time.time() - t0
    logger.info("=== Hybrid Summary ===")
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Text: {text[:200]}")
    logger.info(f"Thinker tokens: {num_gen}")
    logger.info(f"Codec tokens: {num_codec} ({codec_filtered.shape[1]} after filter)")
    logger.info(f"Audio duration: {waveform.numel() / 24000:.2f}s")
    logger.info(f"Total time: {total:.2f}s")
    logger.info(f"  Loading: {t3 - t_load_start:.2f}s")
    logger.info(f"  Thinker: {t_thinker_end - t_thinker_start:.2f}s")
    logger.info(f"  Talker: {t_talker_end - t_talker_start:.2f}s")
    logger.info(f"  Token2Wav: {t_t2w_end - t_t2w_start:.2f}s")


def run_rtp_mode(args):
    """Run e2e with rtp-llm thinker + C++ engine talker + token2wav."""
    from rtp_llm.omni.models.qwen2_5_omni.talker import Qwen2_5OmniTalker
    from rtp_llm.omni.models.qwen2_5_omni.token2wav_model import Token2WavModel
    from rtp_llm.async_decoder_engine.engine_creator import create_engine
    from rtp_llm.ops import (
        ParallelismConfig, RuntimeConfig, FMHAConfig, DeviceResourceConfig,
        MoeConfig, NcclCommConfig, PDSepConfig, ConcurrencyConfig,
        ProfilingDebugLoggingConfig, HWKernelConfig, ModelSpecificConfig,
        SpeculativeExecutionConfig, CacheStoreConfig, MiscellaneousConfig,
        ArpcConfig, GrpcConfig,
    )
    from rtp_llm.config.kv_cache_config import KVCacheConfig
    from rtp_llm.config.py_config_modules import ServerConfig, LoadConfig
    from rtp_llm.config.engine_config import EngineConfig
    import inspect

    device = args.device
    t0 = time.time()

    # Create engine config — use negative start_port so gRPC server won't bind
    talker_server_config = ServerConfig()
    talker_server_config.start_port = -100

    engine_config = EngineConfig(
        parallelism_config=ParallelismConfig(),
        runtime_config=RuntimeConfig(),
        nccl_comm_config=NcclCommConfig(),
        server_config=talker_server_config,
        pd_sep_config=PDSepConfig(),
        concurrency_config=ConcurrencyConfig(),
        fmha_config=FMHAConfig(),
        kv_cache_config=KVCacheConfig(),
        profiling_debug_logging_config=ProfilingDebugLoggingConfig(),
        hw_kernel_config=HWKernelConfig(),
        device_resource_config=DeviceResourceConfig(),
        moe_config=MoeConfig(),
        model_specific_config=ModelSpecificConfig(),
        sp_config=SpeculativeExecutionConfig(),
        cache_store_config=CacheStoreConfig(),
        misc_config=MiscellaneousConfig(),
        arpc_config=ArpcConfig(),
        grpc_config=GrpcConfig(),
        load_config=LoadConfig(),
    )

    # Load talker model with Python model
    logger.info("=== Loading talker engine ===")
    t_load_start = time.time()
    talker_config = Qwen2_5OmniTalker._create_config(args.ckpt)
    talker_config.ckpt_path = args.ckpt
    talker_config.tokenizer_path = args.ckpt
    talker_config.model_type = "qwen2_5_omni_talker"
    talker_config.max_seq_len = 2048
    talker_config.use_kvcache = True
    talker_config.phy2log_path = ""
    talker_config.init_precision_config(
        kv_cache_config=engine_config.kv_cache_config, act_type=None
    )

    from_config_kwargs = dict(
        model_config=talker_config,
        parallelism_config=engine_config.parallelism_config,
        hw_kernel_config=engine_config.hw_kernel_config,
        kv_cache_config=engine_config.kv_cache_config,
        fmha_config=engine_config.fmha_config,
        moe_config=engine_config.moe_config,
        load_method=engine_config.load_config.load_method,
        max_generate_batch_size=engine_config.runtime_config.max_generate_batch_size,
        vit_config=None, merge_lora=False,
        device_resource_config=engine_config.device_resource_config,
        force_cpu_load_weights=engine_config.load_config.force_cpu_load_weights,
    )
    sig = inspect.signature(Qwen2_5OmniTalker.from_config)
    if 'load_python_model' in sig.parameters:
        from_config_kwargs['load_python_model'] = True
    if 'skip_python_model' in sig.parameters:
        from_config_kwargs['skip_python_model'] = False

    talker_model = Qwen2_5OmniTalker.from_config(**from_config_kwargs)

    # Create and start the C++ engine
    alog_conf_path = engine_config.profiling_debug_logging_config.ft_alog_conf_path
    talker_engine = create_engine(
        model=talker_model, engine_config=engine_config,
        alog_conf_path=alog_conf_path, world_info=None,
    )
    talker_engine.start()

    t1 = time.time()
    logger.info(f"Talker C++ engine loaded and started in {t1 - t_load_start:.1f}s")

    # Load token2wav
    token2wav = Token2WavModel.from_pretrained(args.ckpt, device=device)
    t2 = time.time()
    logger.info(f"Token2Wav loaded in {t2 - t1:.1f}s")

    # Load speaker data
    spk_dict = torch.load(os.path.join(args.ckpt, "spk_dict.pt"), map_location=device)
    speaker_data = spk_dict[args.speaker]
    speaker_bos = speaker_data["bos_token"]
    cond = speaker_data["cond"].float().to(device)
    ref_mel = speaker_data["ref_mel"].float().to(device)
    logger.info(f"Speaker: {args.speaker}, BOS: {speaker_bos}")

    # Run thinker via API
    logger.info("=== Running thinker ===")
    t_thinker_start = time.time()
    text, per_token_hs = call_thinker_streaming(args.prompt, args.thinker_url, args.max_thinker_tokens)
    t_thinker_end = time.time()
    logger.info(f"Thinker: {t_thinker_end - t_thinker_start:.2f}s, {len(per_token_hs)} tokens")

    if not per_token_hs:
        logger.error("No hidden states from thinker!")
        return

    # Set thinker hidden states on the Python model
    dtype = torch.bfloat16
    thinker_hs = torch.tensor(per_token_hs, dtype=dtype, device=device)
    logger.info(f"Thinker hidden states: {thinker_hs.shape}")

    py_model = talker_model.py_model
    if py_model is not None and hasattr(py_model, 'set_thinker_hidden_states'):
        py_model.set_thinker_hidden_states(thinker_hs)

    # Run talker via C++ engine generate
    logger.info("=== Running talker (C++ engine) ===")
    initial_tokens = torch.tensor([speaker_bos], dtype=torch.int32)

    t_talker_start = time.time()
    codec_tokens, _ = talker_engine.rtp_llm_op_.generate(
        initial_tokens, args.max_talker_tokens, 8294
    )
    t_talker_end = time.time()
    num_codec = codec_tokens.shape[1]
    logger.info(f"Talker: {t_talker_end - t_talker_start:.2f}s, {num_codec} codec tokens")

    if py_model is not None and hasattr(py_model, 'clear_thinker_hidden_states'):
        py_model.clear_thinker_hidden_states()

    if num_codec == 0:
        logger.error("No codec tokens generated!")
        return

    # Filter special tokens
    mask = codec_tokens[0] < 8292
    codec_filtered = codec_tokens[0][mask].unsqueeze(0)
    if codec_filtered.shape[1] == 0:
        logger.error("All codec tokens were special. No audio.")
        return
    logger.info(f"Codec tokens after filter: {codec_filtered.shape[1]}")

    # Run token2wav
    logger.info("=== Running token2wav ===")
    t_t2w_start = time.time()
    with torch.no_grad():
        waveform = token2wav(codec_filtered.to(device), conditioning=cond, reference_mel=ref_mel)
    t_t2w_end = time.time()

    save_wav(waveform, args.output)

    total = time.time() - t0
    logger.info("=== RTP Engine Summary ===")
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Text: {text[:200]}")
    logger.info(f"Thinker tokens: {len(per_token_hs)}")
    logger.info(f"Codec tokens: {num_codec} ({codec_filtered.shape[1]} after filter)")
    logger.info(f"Audio duration: {waveform.numel() / 24000:.2f}s")
    logger.info(f"Total time: {total:.2f}s")
    logger.info(f"  Loading: {t2 - t_load_start:.2f}s")
    logger.info(f"  Thinker: {t_thinker_end - t_thinker_start:.2f}s")
    logger.info(f"  Talker: {t_talker_end - t_talker_start:.2f}s")
    logger.info(f"  Token2Wav: {t_t2w_end - t_t2w_start:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni end-to-end benchmark")
    parser.add_argument("--ckpt", required=True, help="Model checkpoint path")
    parser.add_argument("--mode", choices=["hf", "hybrid", "rtp"], default="rtp")
    parser.add_argument("--thinker-url", default="http://localhost:18080")
    parser.add_argument("--prompt", default="Tell me a short joke.")
    parser.add_argument("--speaker", default="Chelsie")
    parser.add_argument("--output", default="output.wav")
    parser.add_argument("--max-thinker-tokens", type=int, default=256)
    parser.add_argument("--max-talker-tokens", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.mode == "hf":
        run_hf_mode(args)
    elif args.mode == "rtp":
        run_rtp_mode(args)
    else:
        run_hybrid_mode(args)


if __name__ == "__main__":
    main()
