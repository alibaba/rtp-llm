"""DeepSeek-V4 OpenAI-compatible HTTP server (RTP-LLM integration).

Uses RTP-LLM's OpenAI schema types (`rtp_llm/openai/api_datatype.py`) + the
official V4 `encoding_dsv4.encode_messages` chat template + V4Transformer
backend. Exposes `/v1/chat/completions` compatible with OpenAI clients.

This is a minimal-batching ONE-REQUEST-AT-A-TIME server. Full engine
integration (batching, framework KV cache, PD disagg) is later work — but
this DOES prove V4 serving through RTP-LLM's API surface.

Run:
    export CKPT_DIR=/home/wangyin.yx/.cache/huggingface/hub/.../snapshots/.../
    PYTHONPATH=. /opt/conda310/bin/python -m rtp_llm.models_py.modules.dsv4.serve \\
        --host 0.0.0.0 --port 8000

Client:
    curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
      "messages": [{"role":"user","content":"What is the capital of France?"}],
      "max_tokens": 40, "temperature": 0
    }'
"""

import argparse
import logging
import os
import sys
import threading
import time
import uuid
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

# Import RTP-LLM's OpenAI schema
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatCompletionResponseStreamChoice,
    ChatMessage,
    DeltaMessage,
    FinisheReason,
    RoleEnum,
    UsageInfo,
)

from rtp_llm.models_py.modules.dsv4.test.full_model_generate import load_full_model, CKPT_DIR


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dsv4-serve")


def _load_encoder():
    """Import DeepSeek V4's official chat encoding."""
    enc_dir = os.path.join(CKPT_DIR, "encoding")
    if enc_dir not in sys.path:
        sys.path.insert(0, enc_dir)
    from encoding_dsv4 import encode_messages  # type: ignore
    return encode_messages


class V4Server:
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        log.info("Loading V4-Flash ckpt to %s (this takes ~60s)...", device)
        self.model, self.cfg = load_full_model(device=device)
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(CKPT_DIR, trust_remote_code=True)
        self.encode_messages = _load_encoder()
        # One-request-at-a-time lock — V4 KV cache is per-module register_buffer
        # and would get corrupted under concurrent requests.
        self.lock = threading.Lock()
        log.info("V4-Flash loaded, ready to serve.")

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> int:
        if temperature <= 1e-5:
            return int(logits.argmax().item())
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            cutoff_mask = cumsum > top_p
            cutoff_mask[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(cutoff_mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            # Gumbel-max on truncated set
            sampled = sorted_probs.div(torch.empty_like(sorted_probs).exponential_(1)).argmax(dim=-1)
            return int(sorted_idx[sampled].item())
        return int(probs.div(torch.empty_like(probs).exponential_(1)).argmax(dim=-1).item())

    def _generate_once(self, prompt_ids: List[int], max_tokens: int,
                        temperature: float, top_p: float, stop_ids: List[int]):
        """Generator yielding (token_id, delta_text) one step at a time."""
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        start_pos = 0
        with torch.inference_mode():
            # Prefill
            logits = self.model(input_ids, start_pos=0)
            next_id = self._sample(logits[0], temperature, top_p)
            yield next_id, self.tok.decode([next_id])
            start_pos = input_ids.size(1)
            # Decode
            for _ in range(max_tokens - 1):
                if next_id in stop_ids:
                    break
                cur = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                logits = self.model(cur, start_pos=start_pos)
                next_id = self._sample(logits[0], temperature, top_p)
                yield next_id, self.tok.decode([next_id])
                start_pos += 1

    def _build_prompt_ids(self, messages: List[ChatMessage]) -> List[int]:
        # Convert ChatMessage list to encoding_dsv4 input format
        msg_list = []
        for m in messages:
            role = m.role.value if isinstance(m.role, RoleEnum) else str(m.role)
            # content could be str or list of parts; for now only accept plain str
            if isinstance(m.content, str):
                msg_list.append({"role": role, "content": m.content})
            elif m.content is None:
                msg_list.append({"role": role, "content": ""})
            else:
                # Join text parts
                parts = [p.text for p in m.content if getattr(p, "text", None)]
                msg_list.append({"role": role, "content": "".join(parts)})
        prompt_text = self.encode_messages(msg_list, thinking_mode="chat")
        return self.tok.encode(prompt_text)

    def handle_chat(self, req: ChatCompletionRequest) -> ChatCompletionResponse:
        with self.lock:
            t0 = time.time()
            prompt_ids = self._build_prompt_ids(req.messages)
            max_tokens = req.max_tokens or 256
            temperature = req.temperature if req.temperature is not None else 0.7
            top_p = req.top_p if req.top_p is not None else 1.0
            eos = self.tok.eos_token_id
            stop_ids = [eos] if eos is not None else []

            gen_ids: List[int] = []
            for tok_id, _ in self._generate_once(prompt_ids, max_tokens, temperature, top_p, stop_ids):
                gen_ids.append(tok_id)

            # Strip trailing EOS for the text output
            clean_ids = gen_ids[:-1] if gen_ids and gen_ids[-1] in stop_ids else gen_ids
            text = self.tok.decode(clean_ids, skip_special_tokens=False)
            finish = FinisheReason.stop if (gen_ids and gen_ids[-1] in stop_ids) else FinisheReason.length

            elapsed = time.time() - t0
            log.info("chat: prompt=%d tok, gen=%d tok in %.2fs (%.2f tok/s) -> %r",
                     len(prompt_ids), len(gen_ids), elapsed, len(gen_ids) / elapsed,
                     text[:80] + ("..." if len(text) > 80 else ""))

            return ChatCompletionResponse(
                id=f"chat-{uuid.uuid4().hex[:12]}",
                model=req.model or "deepseek-v4-flash",
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=ChatMessage(role=RoleEnum.assistant, content=text),
                        finish_reason=finish,
                    )
                ],
                usage=UsageInfo(
                    prompt_tokens=len(prompt_ids),
                    completion_tokens=len(gen_ids),
                    total_tokens=len(prompt_ids) + len(gen_ids),
                ),
            )

    def handle_chat_stream(self, req: ChatCompletionRequest):
        """SSE streaming generator."""
        with self.lock:
            prompt_ids = self._build_prompt_ids(req.messages)
            max_tokens = req.max_tokens or 256
            temperature = req.temperature if req.temperature is not None else 0.7
            top_p = req.top_p if req.top_p is not None else 1.0
            eos = self.tok.eos_token_id
            stop_ids = [eos] if eos is not None else []

            resp_id = f"chat-{uuid.uuid4().hex[:12]}"
            model_name = req.model or "deepseek-v4-flash"

            # Initial role frame
            first = ChatCompletionStreamResponse(
                id=resp_id, model=model_name,
                choices=[ChatCompletionResponseStreamChoice(
                    index=0, delta=DeltaMessage(role=RoleEnum.assistant))],
            )
            yield f"data: {first.model_dump_json()}\n\n"

            for tok_id, delta_text in self._generate_once(
                prompt_ids, max_tokens, temperature, top_p, stop_ids,
            ):
                if tok_id in stop_ids:
                    break
                chunk = ChatCompletionStreamResponse(
                    id=resp_id, model=model_name,
                    choices=[ChatCompletionResponseStreamChoice(
                        index=0, delta=DeltaMessage(content=delta_text))],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

            final = ChatCompletionStreamResponse(
                id=resp_id, model=model_name,
                choices=[ChatCompletionResponseStreamChoice(
                    index=0, delta=DeltaMessage(),
                    finish_reason=FinisheReason.stop)],
            )
            yield f"data: {final.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"


def build_app(server: V4Server) -> FastAPI:
    app = FastAPI(title="RTP-LLM DeepSeek-V4 Server")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "deepseek-v4-flash"}

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{"id": "deepseek-v4-flash", "object": "model",
                      "created": int(time.time()), "owned_by": "local"}],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        try:
            if req.stream:
                return StreamingResponse(
                    server.handle_chat_stream(req),
                    media_type="text/event-stream",
                )
            return server.handle_chat(req)
        except Exception as e:
            log.exception("chat_completions failed")
            raise HTTPException(status_code=500, detail=str(e))

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    server = V4Server(device=args.device)
    app = build_app(server)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
