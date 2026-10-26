"""Token-only client for the native inference RPC.

This module imports neither Torch nor RTP native libraries. The generated proto
bindings and grpcio are its only runtime dependencies. Weight lifecycle and
model-version control are deliberately separate from the token transport.
"""

import asyncio
import math
import struct
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import grpc

from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub


class TokenProtocolError(RuntimeError):
    """A reply cannot safely be used as a training sample."""


def _tensor_values(value, dtype: int):
    if value.data_type != dtype:
        raise TokenProtocolError(
            f"Unexpected tensor dtype {value.data_type}, expected {dtype}"
        )
    if not value.shape or any(dim < 0 for dim in value.shape):
        raise TokenProtocolError(f"Invalid tensor shape {list(value.shape)}")
    count = math.prod(value.shape)
    data = value.int32_data if dtype == pb.TensorPB.INT32 else value.fp32_data
    if len(data) != count * 4:
        raise TokenProtocolError("Tensor byte count does not match its shape")
    return list(
        struct.unpack(f"<{count}{'i' if dtype == pb.TensorPB.INT32 else 'f'}", data)
    )


@dataclass(frozen=True)
class TokenSamplingParams:
    max_new_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_new_tokens: int = 0
    seed: Optional[int] = None
    stop_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False
    logprobs: bool = True
    # softmax_probs is computed from model logits BEFORE sampling processors.
    logprobs_mode: str = "raw_logprobs"

    def validate(self):
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if (
            type(self.min_new_tokens) is not int
            or not 0 <= self.min_new_tokens <= self.max_new_tokens
        ):
            raise ValueError("min_new_tokens must lie within the generation budget")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if type(self.top_k) is not int or self.top_k < -1:
            raise ValueError("top_k must be -1, 0, or positive")
        if self.seed is not None and (
            type(self.seed) is not int or not 0 <= self.seed < 2**31
        ):
            # Native GenerateConfig currently stores optional<int>, even though
            # the protobuf wrapper is wider. Never silently truncate a seed.
            raise ValueError("seed must be a nonnegative signed int32")
        if self.logprobs_mode not in ("raw_logprobs", "processed_logprobs"):
            raise ValueError("Unsupported logprobs_mode")
        if (
            self.logprobs
            and self.logprobs_mode == "processed_logprobs"
            and (
                self.temperature not in (0, 1)
                or self.top_p != 1
                or self.top_k not in (-1, 0)
                or self.min_new_tokens != 0
            )
        ):
            raise NotImplementedError(
                "Processed logprobs require temperature=0 or 1, top_p=1, top_k disabled, min_new_tokens=0"
            )
        # vLLM V1's greedy path reports log_softmax of the unscaled logits,
        # not a point mass at the argmax. With all other processors disabled,
        # RTP's FP32 raw softmax therefore also satisfies processed_logprobs.


@dataclass(frozen=True)
class TokenResult:
    token_ids: list[int]
    log_probs: Optional[list[float]]
    finish_reason: str
    logprobs_mode: str
    model_version: str = ""
    global_step: int = 0


class TokenRpcClient:
    def __init__(
        self,
        address: str,
        *,
        vocab_size: int,
        eos_token_ids: Sequence[int],
        timeout: float = 120,
    ):
        if vocab_size <= 0 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("vocab_size and timeout must be positive")
        self.address = address
        self.vocab_size = vocab_size
        self.eos_token_ids = frozenset(eos_token_ids)
        self.timeout = timeout
        self._check_tokens(list(eos_token_ids), allow_empty=True)
        self._channel = None
        self._active = set()
        self._request_ids = set()

    def _check_tokens(self, ids, *, allow_empty=False):
        if not isinstance(ids, (list, tuple)) or (not ids and not allow_empty):
            raise ValueError("A nonempty list of token IDs is required")
        if any(
            type(token) is not int or not 0 <= token < self.vocab_size for token in ids
        ):
            raise ValueError("Token IDs must be integers within the model vocabulary")

    async def close(self):
        for call in tuple(self._active):
            call.cancel()
        if self._channel is not None:
            await self._channel.close()
            self._channel = None

    async def _request(self, request_id: int, ids, config):
        if type(request_id) is not int or not 0 <= request_id < 2**63:
            raise ValueError("request_id must be a nonnegative signed int64")
        if request_id in self._request_ids:
            raise ValueError("request_id is already in flight")
        self._check_tokens(ids)
        if self._channel is None:
            self._channel = grpc.aio.insecure_channel(
                self.address,
                options=[
                    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                ],
            )
        config.timeout_ms = max(1, int(self.timeout * 1000))
        # Older schemas use GenerateConfig's native aux_info=True default.
        if "aux_info" in config.DESCRIPTOR.fields_by_name:
            config.aux_info.value = True
        request = pb.GenerateInputPB(
            request_id=request_id,
            token_ids=ids,
            generate_config=config,
            client_id="verl-token-rollout",
            start_time=int(time.time() * 1e6),
        )
        call = RpcServiceStub(self._channel).GenerateStreamCall(
            request, timeout=self.timeout
        )
        self._active.add(call)
        self._request_ids.add(request_id)
        final = None
        try:
            async for response in call:
                if response.request_id != request_id:
                    raise TokenProtocolError(
                        "Response request_id does not match the request"
                    )
                if (
                    response.HasField("error_info")
                    and response.error_info.error_code != 0
                ):
                    raise TokenProtocolError(str(response.error_info))
                if final is not None:
                    raise TokenProtocolError("Received data after a completed response")
                output = response.flatten_output
                if list(output.finished) != [True]:
                    raise TokenProtocolError(
                        "Expected one completed nonstreaming sequence"
                    )
                final = response
            if final is None:
                raise TokenProtocolError("Empty generation stream")
            return final
        except asyncio.CancelledError:
            call.cancel()
            raise
        finally:
            if not call.done():
                call.cancel()
            self._active.discard(call)
            self._request_ids.discard(request_id)

    def _config(self, params: TokenSamplingParams):
        params.validate()
        self._check_tokens(list(params.stop_token_ids), allow_empty=True)
        config = pb.GenerateConfigPB(
            max_new_tokens=params.max_new_tokens,
            num_beams=1,
            num_return_sequences=1,
            top_k=1 if params.temperature == 0 else max(0, params.top_k),
            top_p=1.0 if params.temperature == 0 else params.top_p,
            temperature=params.temperature or 1.0,
            repetition_penalty=1.0,
            do_sample=params.temperature != 0,
            min_new_tokens=params.min_new_tokens,
            ignore_eos=params.ignore_eos,
            return_incremental=False,
            is_streaming=False,
            return_softmax_probs=params.logprobs,
            reuse_cache=False,
            enable_device_cache=False,
            enable_memory_cache=False,
            enable_remote_cache=False,
            force_disable_sp_run=True,
        )
        for token in params.stop_token_ids:
            config.stop_words_list.rows.add().values.append(token)
        if params.seed is not None:
            config.random_seed.value = params.seed
        return config

    async def generate(
        self,
        request_id: int,
        prompt_ids: list[int],
        params: TokenSamplingParams,
        *,
        expected_version: Optional[str] = None,
    ) -> TokenResult:
        response = await self._request(request_id, prompt_ids, self._config(params))
        if expected_version is not None and response.model_version != expected_version:
            raise TokenProtocolError(
                "Engine model version does not match the requested policy"
            )
        final = response.flatten_output
        ids = _tensor_values(final.output_ids, pb.TensorPB.INT32)
        self._check_tokens(ids)
        if len(final.aux_info) != 1 or final.aux_info[0].input_len != len(prompt_ids):
            raise TokenProtocolError(
                "Response prompt metadata does not match the request"
            )
        info = final.aux_info[0]
        if (
            len(ids) != info.output_len
            or not max(1, params.min_new_tokens) <= len(ids) <= params.max_new_tokens
        ):
            raise TokenProtocolError(
                "Response length is inconsistent with the generation budget"
            )
        scores = None
        if params.logprobs:
            if "raw_logprobs" in info.DESCRIPTOR.fields_by_name and info.HasField(
                "raw_logprobs"
            ):
                scores = _tensor_values(info.raw_logprobs, pb.TensorPB.FP32)
                if len(scores) != len(ids) or any(
                    not math.isfinite(p) or p > 0 for p in scores
                ):
                    raise TokenProtocolError("Invalid selected-token raw logprobs")
            elif not info.HasField("softmax_probs"):
                raise TokenProtocolError("Missing selected-token probabilities")
            else:
                probabilities = _tensor_values(info.softmax_probs, pb.TensorPB.FP32)
                if len(probabilities) != len(ids) or any(
                    not math.isfinite(p) or not 0 < p <= 1 for p in probabilities
                ):
                    raise TokenProtocolError(
                        "Invalid selected-token probabilities; zero/underflow cannot be used as logprobs"
                    )
                # Backward compatibility for engines without the raw FP32 field.
                scores = [math.log(value) for value in probabilities]
        stop_ids = set(params.stop_token_ids) | (
            set() if params.ignore_eos else self.eos_token_ids
        )
        if ids[-1] in stop_ids:
            reason = "stop"
        elif len(ids) == params.max_new_tokens:
            reason = "length"
        else:
            raise TokenProtocolError(
                "Generation ended without EOS, stop token, or length limit"
            )
        return TokenResult(
            ids,
            scores,
            reason,
            params.logprobs_mode,
            response.model_version,
            response.global_step,
        )

    async def score_tokens(
        self,
        request_id: int,
        prompt_ids: list[int],
        response_ids: list[int],
        *,
        expected_version: Optional[str] = None,
    ) -> list[float]:
        self._check_tokens(prompt_ids)
        self._check_tokens(response_ids)
        start = len(prompt_ids) - 1
        end = start + len(response_ids)
        config = self._config(TokenSamplingParams(max_new_tokens=1, logprobs=False))
        # Reserve one discarded generation step explicitly. Some engine branches
        # finish a zero-token request without emitting its prompt logits.
        config.return_prompt_logits = True
        config.return_target_logprob = True
        config.prompt_logits_top_k = 1
        config.prompt_logits_start = start
        config.prompt_logits_end = end
        response = await self._request(request_id, prompt_ids + response_ids, config)
        if expected_version is not None and response.model_version != expected_version:
            raise TokenProtocolError(
                "Engine model version does not match the requested policy"
            )
        final = response.flatten_output
        output = final.prompt_logits
        if not final.HasField("prompt_logits") or (
            output.start_pos,
            output.end_pos,
        ) != (start, end):
            raise TokenProtocolError(
                "Prompt scores do not cover the requested next-token positions"
            )
        values = _tensor_values(output.target_logprobs, pb.TensorPB.FP32)
        if len(values) != len(response_ids) or any(
            not math.isfinite(v) or v > 1e-5 for v in values
        ):
            raise TokenProtocolError("Missing or invalid target logprobs")
        return values
