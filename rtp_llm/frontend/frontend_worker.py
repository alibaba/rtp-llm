import asyncio
import json
import logging
import os
import pathlib
import sys
import time
import traceback
import uuid
from contextvars import ContextVar
from functools import partial
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple, Union

current_file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(current_file_path.parent.absolute()))

from dataclasses import asdict

from pydantic import BaseModel

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.distribute.distributed_server import (
    WorldInfo,
    get_dp_addrs_from_world_info,
    get_world_info,
)
from rtp_llm.frontend.tokenizer_factory.tokenizer_factory import TokenizerFactory
from rtp_llm.models.downstream_modules.kev import KevRequestError
from rtp_llm.ops import ParallelismConfig, SpecialTokens, VitSeparation
from rtp_llm.pipeline.pipeline import Pipeline
from rtp_llm.structure.request_extractor import Request, RequestExtractor
from rtp_llm.utils.base_model_datatypes import (
    BatchedTerminalOutputs,
    CustomOutput,
    GenerateResponse,
)
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)
from rtp_llm.utils.prompt_logits_utils import build_prompt_logits_dict


class PipelineResponse(BaseModel):
    response: str = ""
    finished: bool = True
    aux_info: Dict[str, Any] = {}
    hidden_states: Optional[Union[List[float], List[List[float]]]] = None
    loss: Optional[Union[float, List[float]]] = None
    logits: Optional[Union[List[float], List[List[float]]]] = None
    output_ids: Optional[List[List[int]]] = None
    input_ids: Optional[List[List[int]]] = None
    prompt_logprobs: Optional[Dict[str, Any]] = None
    custom_output: Optional[CustomOutput] = None


class MultiSequencesPipelineResponse(BaseModel):
    response: List[str]
    finished: bool
    aux_info: List[Dict[str, Any]] = {}
    custom_output: Optional[List[Optional[CustomOutput]]] = None


class BatchPipelineResponse(BaseModel):
    response_batch: List[Union[PipelineResponse, MultiSequencesPipelineResponse]]


class TokenizerEncodeResponse(BaseModel):
    token_ids: List[int] = []
    offset_mapping: Optional[List[Any]] = None
    tokens: List[str] = []
    error: str = ""


def d3_model_name(ckpt_path: str) -> str:
    """The served model name of a d3 checkpoint: its manifest name, else the directory."""
    manifest_path = pathlib.Path(ckpt_path) / "MODEL_MANIFEST.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except ValueError:
            manifest = None
        name = manifest.get("model_name") if isinstance(manifest, dict) else None
        if isinstance(name, str) and name.strip():
            return name
    return pathlib.Path(ckpt_path).name


def d3_executor(
    ckpt_path: str,
    run_rows,
    *,
    tokenizer,
    max_seq_len: Optional[int],
    kev_config,
    jev_config,
    timeout_seconds: float | None,
):
    """Build the SystemOne D3 executor of a d3 checkpoint, or None for other models.

    A checkpoint may carry at most one of the Kev, JEV and D3 task manifests.
    """
    from rtp_llm.models.downstream_modules.d3_config import load_d3_config

    config = load_d3_config(ckpt_path)
    if config is None:
        return None
    if kev_config is not None or jev_config is not None:
        raise ValueError("Checkpoint cannot combine D3 with Kev or JEV manifests")
    from rtp_llm.frontend.d3 import D3Executor

    max_rows = int(os.environ.get("D3_MAX_ROWS_PER_RPC", "8"))
    if not 1 <= max_rows <= 64:
        raise ValueError("D3_MAX_ROWS_PER_RPC must be in 1..64")
    return D3Executor(
        config,
        run_rows,
        tokenizer=tokenizer,
        model_name=d3_model_name(ckpt_path),
        max_row_tokens=max_seq_len,
        batch_rows=max_rows,
        timeout_seconds=timeout_seconds,
    )


class FrontendWorker:
    def __init__(
        self,
        py_env_configs,
        model_config,
        special_tokens,
    ) -> None:
        logging.info("starting frontend worker")

        self.tokenizer = TokenizerFactory.create(
            model_config.ckpt_path, model_config.tokenizer_path, model_config.model_type
        )

        # Create engine_config from config (ports from server_config/distribute_config)
        engine_config = EngineConfig.create(py_env_configs, nccl_comm_config=None)

        # Get world_info from config
        world_info = get_world_info(
            server_config=py_env_configs.server_config,
            distribute_config=py_env_configs.distribute_config,
            parallelism_config=py_env_configs.parallelism_config,
        )

        # Get addresses from distribute_info
        addresses = get_dp_addrs_from_world_info(
            world_info=world_info,
            parallelism_config=engine_config.parallelism_config,
        )

        vit_separation = None
        if py_env_configs.vit_config:
            vit_separation = py_env_configs.vit_config.vit_separation

        self.pipeline = Pipeline(
            special_tokens=special_tokens,
            pd_sep_config=engine_config.pd_sep_config,
            addresses=addresses,
            max_seq_len=model_config.max_seq_len,
            seq_size_per_block=model_config.attn_config.tokens_per_block,
            tokenizer=self.tokenizer,
            sp_config=py_env_configs.sp_config,
            mm_related_params=None,  # Frontend doesn't need mm_related_params
            grpc_config=py_env_configs.grpc_config,
            vit_separation=vit_separation,
            mm_model_config=getattr(model_config, "mm_model_config", None),
            server_config=py_env_configs.server_config,
            master_config=py_env_configs.master_config,
            parallelism_config=engine_config.parallelism_config,
            prefill_cp_config=py_env_configs.prefill_cp_config,
        )
        self.backend_rpc_server_visitor = self.pipeline.backend_rpc_server_visitor
        self.generate_env_config = py_env_configs.generate_env_config
        self.server_config = py_env_configs.server_config

        from rtp_llm.models.downstream_modules.kev import load_kev_config

        self.kev_config = load_kev_config(model_config.ckpt_path)
        self.kev_max_row_tokens = model_config.max_seq_len
        self.kev_reuse_cache = bool(py_env_configs.kv_cache_config.reuse_cache)
        self.kev_shared_prefix = os.environ.get("KEV_SHARED_PREFIX", "0") == "1"
        self.kev_tokens_per_block = model_config.attn_config.tokens_per_block

        from rtp_llm.frontend.jev import JevExecutor, JevHeadSpec
        from rtp_llm.models.downstream_modules.jev import load_jev_config

        self.jev_config = load_jev_config(model_config.ckpt_path)
        self.jev_executor = None
        self._jev_request_context = ContextVar("jev_request_context")
        if self.jev_config is not None:
            if self.kev_config is not None:
                raise ValueError("Checkpoint cannot contain both JEV and Kev manifests")
            max_rows = int(os.environ.get("JEV_MAX_CONCURRENT_ROWS", "4"))
            if not 1 <= max_rows <= 64:
                raise ValueError("JEV_MAX_CONCURRENT_ROWS must be in 1..64")
            self.jev_executor = JevExecutor(
                JevHeadSpec.from_config(self.jev_config),
                self._jev_row,
                max_concurrency=max_rows,
                timeout_seconds=(
                    engine_config.pd_sep_config.max_rpc_timeout_ms / 1000
                    if engine_config.pd_sep_config.max_rpc_timeout_ms > 0
                    else 300
                ),
            )

        self.d3_executor = d3_executor(
            model_config.ckpt_path,
            self._d3_rows,
            tokenizer=self.tokenizer,
            max_seq_len=model_config.max_seq_len,
            kev_config=self.kev_config,
            jev_config=self.jev_config,
            timeout_seconds=(
                engine_config.pd_sep_config.max_rpc_timeout_ms / 1000
                if engine_config.pd_sep_config.max_rpc_timeout_ms > 0
                else 300
            ),
        )
        self._d3_request_context = ContextVar("d3_request_context")

        logging.info("frontend worker start done.")

    async def close(self):
        await self.pipeline.close()

    def stop(self):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.close())
        else:
            loop.create_task(self.close())

    def tokenizer_offset_mapping(self, prompt: str) -> Any:
        return self.pipeline.tokenizer(
            prompt, return_offsets_mapping=True, return_attention_mask=False
        )

    def tokenizer_encode(self, prompt: str) -> Tuple[List[int], List[str]]:
        token_ids = self.pipeline.encode(prompt)
        token_ids = [int(id) for id in token_ids]
        tokens = [self.pipeline.decode(id) for id in token_ids]
        return token_ids, tokens

    async def decide(
        self, request: dict, *, headers: Optional[Dict[str, str]] = None
    ) -> dict:
        if getattr(self, "jev_executor", None) is not None:
            return await self._jev_request("decide", request, headers)
        results, _ = await self._decide(request, headers=headers)
        return {"results": results}

    async def systemone(
        self, request: dict, *, headers: Optional[Dict[str, str]] = None
    ) -> dict:
        if getattr(self, "d3_executor", None) is not None:
            return await self._d3_request(request, headers)
        if getattr(self, "jev_executor", None) is not None:
            return await self._jev_request("systemone", request, headers)
        from rtp_llm.frontend.systemone import parse_request, to_answers

        started = time.perf_counter()
        if self.kev_config is None:
            raise KevRequestError("/v1/systemone requires an exported Kev checkpoint")
        try:
            model, record, metadata = parse_request(
                request, max_options=self.kev_config["max_options"]
            )
        except ValueError as error:
            raise KevRequestError(str(error)) from error
        results, input_tokens = await self._decide(record, headers=headers)
        answers = to_answers([result["probabilities"] for result in results], metadata)
        tokenizer = self.tokenizer
        if hasattr(tokenizer, "get_real_tokenizer"):
            tokenizer = tokenizer.get_real_tokenizer()
        # System One counts serialized answer tokens, although no tokens are generated.
        output_tokens = len(
            tokenizer(json.dumps(answers), add_special_tokens=False).input_ids
        )
        return {
            "model": model,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            # Includes frontend processing and RPC queue/execution time.
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    def decide_info(self) -> dict:
        if getattr(self, "jev_executor", None) is None:
            raise KevRequestError("/v1/decide/info requires an exported JEV checkpoint")
        return self.jev_executor.info()

    async def _jev_request(self, method: str, request: dict, headers) -> dict:
        # All rows of one decision route to the same replica. ContextVar keeps
        # concurrent callers' correlation headers and routing keys independent.
        token = self._jev_request_context.set(
            (dict(headers or {}), uuid.uuid4().int & ((1 << 63) - 1))
        )
        try:
            return await getattr(self.jev_executor, method)(request)
        finally:
            self._jev_request_context.reset(token)

    async def _jev_row(self, row):
        import torch

        from rtp_llm.frontend.jev import JevRowResult
        from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
        from rtp_llm.utils.base_model_datatypes import GenerateInput, MMUrlType

        text, images = [], []
        for part in row.content:
            if part["type"] == "text":
                text.append(part["text"])
            else:
                text.append("<|vision_start|><|image_pad|><|vision_end|>")
                images.append(
                    MultimodalInput(
                        part["image_url"]["url"],
                        MMUrlType.IMAGE,
                        torch.empty(0),
                        MMPreprocessConfig(),
                    )
                )
        tokenizer = self.tokenizer
        if hasattr(tokenizer, "get_real_tokenizer"):
            tokenizer = tokenizer.get_real_tokenizer()
        # System1 uses a bare prompt, without the model's chat template/BOS.
        ids = tokenizer("".join(text), add_special_tokens=False).input_ids
        if not ids or len(ids) > self.kev_max_row_tokens:
            raise KevRequestError("JEV row exceeds the configured context length")
        headers, routing_key = self._jev_request_context.get()
        item = GenerateInput(
            request_id=uuid.uuid4().int & ((1 << 63) - 1),
            token_ids=torch.tensor(ids, dtype=torch.int32),
            mm_inputs=images,
            generate_config=GenerateConfig(max_new_tokens=0, is_streaming=False),
            tokenizer=self.tokenizer,
            headers=dict(headers),
        )
        outputs = await self.backend_rpc_server_visitor.batch_enqueue(
            [item], routing_key=routing_key
        )
        if len(outputs) != 1 or len(outputs[0].generate_outputs) != 1:
            raise RuntimeError("JEV backend must return one result per row")
        result = outputs[0].generate_outputs[0]
        logits = result.custom_output
        if (
            not result.finished
            or logits is None
            or logits.shape != (1, self.jev_executor.spec.num_labels)
            or not torch.isfinite(logits).all().item()
        ):
            raise RuntimeError("JEV backend returned invalid decision logits")
        # AuxInfo.input_len is the expanded multimodal length reported by the
        # engine. Each image placeholder becomes that many additional tokens.
        prompt_tokens = result.aux_info.input_len or len(ids)
        vision_tokens = (
            result.aux_info.multimodal_lengths.get(
                int(MMUrlType.IMAGE), max(0, prompt_tokens - len(ids) + len(images))
            )
            if images
            else 0
        )
        return JevRowResult(
            logits=logits[0].tolist(),
            prompt_tokens=prompt_tokens,
            vision_tokens=vision_tokens,
            cache_reuse_tokens=result.aux_info.reuse_len,
        )

    async def _d3_request(self, request: dict, headers) -> dict:
        # Every row of one decision routes to the same replica, with per-request
        # correlation headers kept independent of concurrent callers.
        token = self._d3_request_context.set(
            (dict(headers or {}), uuid.uuid4().int & ((1 << 63) - 1))
        )
        try:
            return await self.d3_executor.systemone(request)
        finally:
            self._d3_request_context.reset(token)

    async def _d3_rows(self, rows):
        """Submit complete d3 rows through one bounded batch RPC, in request order.

        Every row is an independent GenerateInput with its own request id; the
        request's rows share one routing key. No shared-prefix RPC is used: d3's
        noncausal full-attention rows must never reuse language-layer prefixes.
        """
        import torch

        from rtp_llm.frontend.d3 import (
            IMAGE_MAX_PIXELS,
            IMAGE_MIN_PIXELS,
            MAX_OPTIONS,
            D3RowResult,
        )
        from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
        from rtp_llm.utils.base_model_datatypes import GenerateInput, MMUrlType

        headers, routing_key = self._d3_request_context.get()
        inputs = []
        for row in rows:
            mm_inputs = []
            for media in row.media:
                if media.kind == "image":
                    mm_inputs.append(
                        MultimodalInput(
                            media.url,
                            MMUrlType.IMAGE,
                            torch.empty(0),
                            MMPreprocessConfig(
                                min_pixels=IMAGE_MIN_PIXELS, max_pixels=IMAGE_MAX_PIXELS
                            ),
                        )
                    )
                else:
                    # One d3 video group: the envelope URL carries the clip
                    # identity, group index and frame size, the tensor the
                    # group's already-resized uint8 frames. The consumer only
                    # normalizes and patchifies them (no second resize).
                    mm_inputs.append(
                        MultimodalInput(
                            media.url,
                            MMUrlType.VIDEO,
                            media.tensor,
                            MMPreprocessConfig(),
                        )
                    )
            inputs.append(
                GenerateInput(
                    request_id=uuid.uuid4().int & ((1 << 63) - 1),
                    token_ids=torch.tensor(row.token_ids, dtype=torch.int32),
                    mm_inputs=mm_inputs,
                    generate_config=GenerateConfig(
                        max_new_tokens=0, is_streaming=False
                    ),
                    tokenizer=self.tokenizer,
                    headers=dict(headers),
                )
            )
        outputs = await self.backend_rpc_server_visitor.batch_enqueue(
            inputs, routing_key=routing_key
        )
        if len(outputs) != len(inputs):
            raise RuntimeError("D3 backend returned an incorrect row count")
        results = []
        for index, output in enumerate(outputs):
            if len(output.generate_outputs) != 1:
                raise RuntimeError("D3 backend must return one result per row")
            result = output.generate_outputs[0]
            logits = result.custom_output
            # Structural corruption only: an unfinished row, a missing readout
            # or a wrong shape cannot produce any answer and fails the batch
            # (the row RPC is atomic). NON-FINITE logits are deliberately NOT
            # rejected here: the official contract maps them to a per-question
            # invalid_model_output, which the executor's probabilities() already
            # implements (over the question's own masked answer codes, exactly
            # like the official product_answer); rejecting them here would
            # pre-empt that with a whole-request 500.
            if (
                not result.finished
                or logits is None
                or logits.shape != (1, MAX_OPTIONS)
            ):
                raise RuntimeError("D3 backend returned invalid readout logits")
            reuse_len = result.aux_info.reuse_len if result.aux_info else 0
            if reuse_len:
                raise RuntimeError("D3 rows must not reuse cached prefixes")
            # aux_info.input_len is the engine's expanded multimodal length; the
            # placeholder ids left in token_ids are not counted by the engine.
            prompt_tokens = (
                result.aux_info.input_len if result.aux_info else 0
            ) or len(rows[index].token_ids)
            results.append(
                D3RowResult(
                    logits=logits[0].tolist(),
                    prompt_tokens=prompt_tokens,
                    reuse_len=reuse_len,
                )
            )
        return results

    async def _decide(
        self, request: dict, *, headers: Optional[Dict[str, str]] = None
    ) -> tuple[list[dict], int]:
        """Run independent Kev rows through the existing prefill-only batch RPC."""
        import torch

        from rtp_llm.models.downstream_modules.kev import encode_kev_rows
        from rtp_llm.utils.base_model_datatypes import GenerateInput

        if self.kev_config is None:
            raise KevRequestError("/v1/decide requires an exported Kev checkpoint")
        if not isinstance(request, dict) or set(request) != {"state", "questions"}:
            raise KevRequestError("request must contain exactly state and questions")
        questions = request["questions"]
        if not isinstance(questions, list) or len(questions) > 64:
            raise KevRequestError("questions must contain 1..64 entries")
        rows = encode_kev_rows(
            self.tokenizer,
            request["state"],
            questions,
            max_state_tokens=self.kev_config["max_state_tokens"],
            max_row_tokens=min(
                self.kev_max_row_tokens, self.kev_config["max_row_tokens"]
            ),
            max_options=self.kev_config["max_options"],
        )
        state_tokens = rows[0].index(self.kev_config["special_token_ids"][1])
        warm_prefix = (
            self.kev_reuse_cache
            and self.kev_tokens_per_block > 0
            and state_tokens >= self.kev_tokens_per_block
            and len(rows) > 1
        )
        routing_key = uuid.uuid4().int & ((1 << 63) - 1)
        all_inputs = [
            GenerateInput(
                request_id=uuid.uuid4().int & ((1 << 63) - 1),
                token_ids=torch.tensor(row, dtype=torch.int32),
                mm_inputs=[],
                generate_config=GenerateConfig(max_new_tokens=0, is_streaming=False),
                tokenizer=self.tokenizer,
                headers=dict(headers or {}),
            )
            for row in rows
        ]
        shared_outputs = None
        if getattr(self, "kev_shared_prefix", False) and self.kev_reuse_cache:
            from rtp_llm.cpp.model_rpc.model_rpc_client import SharedPrefixUnsupported

            try:
                shared_outputs = (
                    await self.backend_rpc_server_visitor.shared_prefix_batch_enqueue(
                        all_inputs, state_tokens=state_tokens, routing_key=routing_key
                    )
                )
            except SharedPrefixUnsupported:
                # UNIMPLEMENTED is guaranteed to precede enqueue on this RPC.
                # Never retry errors or missing acknowledgements after execution.
                logging.warning(
                    "Selected backend lacks shared-prefix RPC; using legacy Kev submission"
                )
        results = []
        start = 0
        while start < len(rows):
            if shared_outputs is not None:
                end = len(rows)
            else:
                end, tokens = start, 0
                limit = 1 if warm_prefix and start == 0 else len(rows)
                while end < limit and (
                    end == start or tokens + len(rows[end]) <= 16384
                ):
                    tokens += len(rows[end])
                    end += 1
            inputs = all_inputs[start:end]
            outputs = (
                shared_outputs
                if shared_outputs is not None
                else await self.backend_rpc_server_visitor.batch_enqueue(
                    inputs, routing_key=routing_key
                )
            )
            if len(outputs) != len(inputs):
                raise RuntimeError("Kev backend returned an incorrect row count")
            for index, output in enumerate(outputs, start):
                if len(output.generate_outputs) != 1:
                    raise RuntimeError("Kev backend must return one result per row")
                result = output.generate_outputs[0]
                probs = result.custom_output
                if (
                    not result.finished
                    or probs is None
                    or probs.shape != (1, self.kev_config["max_options"])
                    or not torch.isfinite(probs).all().item()
                ):
                    raise RuntimeError("Kev backend returned an invalid decision")
                count = len(questions[index]["options"])
                probs = probs[0, :count]
                if (probs < 0).any().item() or abs(probs.sum().item() - 1) > 1e-5:
                    raise RuntimeError("Kev backend returned invalid probabilities")
                choice = int(probs.argmax().item())
                results.append(
                    {
                        "probabilities": probs.tolist(),
                        "choice": choice,
                        "option": questions[index]["options"][choice],
                    }
                )
            start = end
        # Logical input length counts the shared state once, independent of cache hits.
        input_tokens = sum(map(len, rows)) - (len(rows) - 1) * state_tokens
        return results, input_tokens

    async def batch_infer(
        self, prompts: List[str], request_id: int, generate_config: dict
    ) -> BatchPipelineResponse:
        responses = await self.pipeline.batch_infer(
            prompts=prompts,
            base_request_id=request_id,
            generate_config_json=generate_config,
            generate_env_config=self.generate_env_config,
        )
        # Reconstruct GenerateConfig to check flags (aux_info, calculate_loss, etc.)
        gc = self.pipeline.create_generate_config(
            generate_config,
            len(self.pipeline.tokenizer),
            self.pipeline._special_tokens,
            self.pipeline.tokenizer,
            generate_env_config=self.generate_env_config,
        )
        pipeline_responses = []
        for gen_response in responses:
            out = gen_response.generate_outputs.generate_outputs[0]
            generate_texts = gen_response.generate_texts
            aux_info_dict: Dict[str, Any] = {}
            if gc.aux_info:
                aux = out.aux_info
                if gc.has_num_beams():
                    aux.beam_responses = generate_texts
                aux_info_dict = asdict(aux)
            prompt_logits_dict = (
                build_prompt_logits_dict(out.prompt_logits)
                if gc.return_prompt_logits
                else None
            )
            pipeline_responses.append(
                PipelineResponse(
                    response=generate_texts[0],
                    finished=out.finished,
                    aux_info=aux_info_dict,
                    hidden_states=(
                        out.hidden_states.tolist()
                        if gc.return_hidden_states and out.hidden_states is not None
                        else None
                    ),
                    loss=(
                        out.loss.tolist()
                        if gc.calculate_loss and out.loss is not None
                        else None
                    ),
                    logits=(
                        out.logits.tolist()
                        if gc.return_logits and out.logits is not None
                        else None
                    ),
                    prompt_logprobs=prompt_logits_dict,
                    custom_output=(
                        out.custom_output.tolist()
                        if out.custom_output is not None
                        else None
                    ),
                )
            )
        return BatchPipelineResponse(response_batch=pipeline_responses)

    def inference(self, **kwargs: Any) -> CompleteResponseAsyncGenerator:
        default_generate_config = GenerateConfig()
        request_extractor = RequestExtractor(default_generate_config)
        request, kwargs = request_extractor.extract_request(kwargs)

        if request.is_streaming is False and request.incremental:
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                "request is non_stream but use incremental decoder",
            )

        response_generator = self._inference(request, **kwargs)

        complete_response_collect_func = partial(
            FrontendWorker.collect_complete_response,
            incremental=request.incremental,
            batch_infer=request.batch_infer,
            num_return_sequences=request.num_return_sequences,
        )
        return CompleteResponseAsyncGenerator(
            response_generator, complete_response_collect_func
        )

    def _inference(self, request: Request, **kwargs: Any):
        if (
            len(request.input_texts) > 1
            or request.batch_infer
            or request.num_return_sequences > 0
        ):
            num_return_sequences = request.generate_configs[0].num_return_sequences
            generators: List[AsyncGenerator[Dict[str, Any], None]] = []
            # TODO temp fix sp with batch infer, will change request_id to str later
            group_size = len(request.input_texts)
            # Use request.request_id as group_id for all streams in the same batch
            group_id = request.request_id
            for i, (text, urls, generate_config) in enumerate(
                zip(request.input_texts, request.input_urls, request.generate_configs)
            ):
                generators.append(
                    self._yield_generate(
                        request.request_id + i * 10000,
                        text,
                        urls,
                        generate_config=generate_config,
                        group_size=group_size,
                        group_id=group_id,
                        **kwargs,
                    )
                )
            return self._parallel_batch_async_generators(
                request.incremental,
                generators,
                request.batch_infer,
            )
        else:
            return self._yield_generate(
                request.request_id,
                request.input_texts[0],
                request.input_urls[0],
                generate_config=request.generate_configs[0],
                **kwargs,
            )

    def _format_response(
        self,
        gen_responses: GenerateResponse,
        generate_config: GenerateConfig,
    ) -> Dict[str, Any]:
        generate_texts = gen_responses.generate_texts
        finished = gen_responses.generate_outputs.generate_outputs[0].finished
        if generate_config.aux_info:
            aux_info = gen_responses.generate_outputs.generate_outputs[0].aux_info
            if generate_config.has_num_beams():
                aux_info.beam_responses = generate_texts
        hidden_states = gen_responses.generate_outputs.generate_outputs[0].hidden_states
        output_ids = gen_responses.generate_outputs.generate_outputs[0].output_ids
        input_ids = gen_responses.generate_outputs.generate_outputs[0].input_ids
        loss = gen_responses.generate_outputs.generate_outputs[0].loss
        logits = gen_responses.generate_outputs.generate_outputs[0].logits
        prompt_logits_raw = gen_responses.generate_outputs.generate_outputs[
            0
        ].prompt_logits

        prompt_logits_dict = (
            build_prompt_logits_dict(prompt_logits_raw)
            if generate_config.return_prompt_logits
            else None
        )
        custom_output = gen_responses.generate_outputs.generate_outputs[0].custom_output

        response = PipelineResponse(
            response=generate_texts[0],
            finished=finished,
            aux_info=asdict(aux_info) if generate_config.aux_info else {},
            hidden_states=(
                hidden_states.tolist()
                if generate_config.return_hidden_states and hidden_states is not None
                else None
            ),
            loss=(
                loss.tolist()
                if generate_config.calculate_loss and loss is not None
                else None
            ),
            logits=(
                logits.tolist()
                if generate_config.return_logits and logits is not None
                else None
            ),
            output_ids=(
                output_ids.tolist()
                if generate_config.return_output_ids and output_ids is not None
                else None
            ),
            input_ids=(
                input_ids.tolist()
                if generate_config.return_input_ids and input_ids is not None
                else None
            ),
            prompt_logprobs=prompt_logits_dict,
            custom_output=(
                custom_output.tolist() if custom_output is not None else None
            ),
        )

        return response

    def _format_response_new(
        self,
        gen_responses: GenerateResponse,
        generate_config: GenerateConfig,
    ) -> Dict[str, Any]:
        generate_texts = gen_responses.generate_texts
        if generate_config.num_return_sequences > 0:
            outputs = gen_responses.generate_outputs.generate_outputs
            custom_outputs = None
            if not isinstance(outputs, BatchedTerminalOutputs) and any(
                seq.custom_output is not None for seq in outputs
            ):
                custom_outputs = [
                    (
                        seq.custom_output.tolist()
                        if seq.custom_output is not None
                        else None
                    )
                    for seq in outputs
                ]
            aux_info = []
            if generate_config.aux_info:
                aux_info = []
                for seq in gen_responses.generate_outputs.generate_outputs:
                    info = asdict(seq.aux_info)
                    aux_info.append(info)
            sequences_pipeline_response = MultiSequencesPipelineResponse(
                response=generate_texts,
                finished=isinstance(
                    gen_responses.generate_outputs.generate_outputs,
                    BatchedTerminalOutputs,
                )
                or all(
                    [
                        seq.finished
                        for seq in gen_responses.generate_outputs.generate_outputs
                    ]
                ),
                aux_info=aux_info,
                custom_output=custom_outputs,
            )
            return sequences_pipeline_response
        else:
            return self._format_response(gen_responses, generate_config)

    async def _yield_generate(
        self,
        request_id: int,
        text: str,
        urls: List[str],
        generate_config: GenerateConfig,
        group_size: int = 1,
        group_id: int = -1,
        **kwargs: Any,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        stream = self.pipeline.pipeline_async(
            prompt=text,
            request_id=request_id,
            urls=urls,
            generate_config=generate_config,
            generate_env_config=self.generate_env_config,
            group_size=group_size,
            group_id=group_id,
            **kwargs,
        )
        async for generate_response in stream:
            yield self._format_response_new(generate_response, generate_config)

    def is_streaming(self, req: Dict[str, Any]):
        return RequestExtractor.is_streaming(req) or req.get("stream", False)

    async def _parallel_batch_async_generators(
        self,
        incremental: bool,
        generators: List[AsyncGenerator[Dict[str, Any], None]],
        batch_infer: bool,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        iterators = [gen.__aiter__() for gen in generators]
        done_idxs: Set[int] = set()
        batch_state: List[Any] = [None] * len(iterators)

        while True:
            # 创建并行任务
            tasks = []
            for idx, itr in enumerate(iterators):
                if idx not in done_idxs:  # 仅为未完成的迭代器创建任务
                    tasks.append((idx, itr.__anext__()))

            # 使用 asyncio.gather() 获取结果
            if tasks:
                results = await asyncio.gather(
                    *(task[1] for task in tasks), return_exceptions=True
                )
                for idx, result in zip((task[0] for task in tasks), results):
                    if isinstance(result, Exception):
                        # 处理异常情况，如 StopAsyncIteration
                        if isinstance(result, StopAsyncIteration):
                            done_idxs.add(idx)
                            if batch_state[idx] is None:
                                batch_state[idx] = PipelineResponse()
                            if incremental:
                                batch_state[idx] = PipelineResponse()
                        else:
                            error_msg = f'ErrorMsg: {str(result)} \n Traceback: {"".join(traceback.format_tb(result.__traceback__))}'
                            logging.warning(error_msg)
                            raise result
                    else:
                        batch_state[idx] = result

            # 检查是否所有迭代器都完成
            if len(done_idxs) == len(iterators):
                break

            # 处理 batch 数据
            batch = batch_state
            if batch_infer:
                yield BatchPipelineResponse(response_batch=batch)
            else:
                yield batch[0]

    # TODO(xinfei.sxf) 这个函数将逻辑又写了一遍
    @staticmethod
    async def collect_complete_response(
        all_responses: List[
            Union[
                PipelineResponse, MultiSequencesPipelineResponse, BatchPipelineResponse
            ]
        ],
        incremental: bool,
        batch_infer: bool,
        num_return_sequences: int,
    ) -> Union[PipelineResponse, MultiSequencesPipelineResponse, BatchPipelineResponse]:

        if not incremental:
            return await CompleteResponseAsyncGenerator.get_last_value(all_responses)

        if batch_infer:
            batch_response_incremental_stream = None
            async for response in all_responses:
                if not batch_response_incremental_stream:
                    batch_response_incremental_stream = [
                        [_] for _ in response.response_batch
                    ]
                else:
                    for batch_idx, single_response in enumerate(
                        response.response_batch
                    ):
                        batch_response_incremental_stream[batch_idx].append(
                            single_response
                        )
            complete_batch_response = []
            async for (
                single_response_incremental_stream
            ) in CompleteResponseAsyncGenerator.generate_from_list(
                batch_response_incremental_stream
            ):
                single_yield_response = (
                    CompleteResponseAsyncGenerator.generate_from_list(
                        single_response_incremental_stream
                    )
                )
                single_complete_response = (
                    await FrontendWorker.collect_complete_response(
                        single_yield_response, incremental, False, num_return_sequences
                    )
                )
                complete_batch_response.append(single_complete_response)
            return BatchPipelineResponse(response_batch=complete_batch_response)

        if num_return_sequences > 0:
            complete_multi_seq_response = None
            complete_multi_seq_finished = None
            complete_multi_seq_aux_info = None
            complete_multi_seq_custom_output = None
            async for response in all_responses:
                if not complete_multi_seq_response:
                    complete_multi_seq_response = [_ for _ in response.response]
                    complete_multi_seq_aux_info = [_ for _ in response.aux_info]
                    complete_multi_seq_finished = response.finished
                if response.custom_output is not None:
                    complete_multi_seq_custom_output = response.custom_output
                for seq_idx, seq_reponse in enumerate(response.response):
                    complete_multi_seq_response[seq_idx] = (
                        complete_multi_seq_response[seq_idx] + seq_reponse
                    )
                    if response.aux_info and response.aux_info[seq_idx]:
                        complete_multi_seq_aux_info[seq_idx] = response.aux_info[
                            seq_idx
                        ]
                    if response.finished:
                        complete_multi_seq_finished = True
            return MultiSequencesPipelineResponse(
                response=complete_multi_seq_response,
                aux_info=complete_multi_seq_aux_info,
                finished=complete_multi_seq_finished,
                custom_output=complete_multi_seq_custom_output,
            )
        complete_response = ""
        finished = False
        aux_info = None
        output_ids = None
        input_ids = None
        logits = None
        custom_output = None
        async for response in all_responses:
            complete_response = complete_response + response.response
            if response.finished:
                finished = response.finished
            if response.aux_info:
                aux_info = response.aux_info
            if response.output_ids:
                output_ids = response.output_ids
            if response.input_ids:
                input_ids = response.input_ids
            if response.logits:
                logits = response.logits
            if response.custom_output is not None:
                custom_output = response.custom_output
        return PipelineResponse(
            response=complete_response,
            finished=finished,
            aux_info=aux_info,
            output_ids=output_ids,
            input_ids=input_ids,
            logits=logits,
            custom_output=custom_output,
        )
