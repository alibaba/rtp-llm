import hashlib
import logging
import os
import signal
import threading
import time
from array import array
from concurrent import futures
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import grpc
import torch

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import setup_and_configure_server
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    ROLE_TYPE_VIT,
    CacheStatusPB,
    EmptyPB,
    MMPreprocessConfigPB,
    MMRdmaDescPB,
    MultimodalInputsPB,
    MultimodalOutputsPB,
    TensorPB,
    WorkerStatusPB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
    MultimodalRpcServiceServicer,
    add_MultimodalRpcServiceServicer_to_server,
)
from rtp_llm.distribute.distributed_server import get_world_info
from rtp_llm.model_factory import ModelFactory
from rtp_llm.ops import (
    MMRdmaEncoderOp,
    VitSeparation,
    ensure_engine_ops_loaded,
    get_multimodal_feature_hash,
)
from rtp_llm.server.server_args.server_args import setup_args
from rtp_llm.server.vit_token_id_cache import MMTokenIdCache
from rtp_llm.utils.grpc_util import trans_from_tensor, trans_tensor
from rtp_llm.utils.mm_process_engine import MMEmbeddingRes, MMProcessEngine
from rtp_llm.utils.multimodal_util import MMUrlType, url_data_cache_, vit_emb_cache_
from rtp_llm.utils.process_manager import ProcessManager

setup_logging()


def trans_config(mm_process_config_pb: MMPreprocessConfigPB):
    return [
        mm_process_config_pb.width,
        mm_process_config_pb.height,
        mm_process_config_pb.min_pixels,
        mm_process_config_pb.max_pixels,
        mm_process_config_pb.fps,
        mm_process_config_pb.min_frames,
        mm_process_config_pb.max_frames,
        mm_process_config_pb.mm_padding_size,
    ]


def trans_input(mutlimodal_inputs_pb: MultimodalInputsPB):
    urls = []
    types = []
    tensors = []
    configs = []
    try:
        for mm_input in mutlimodal_inputs_pb.multimodal_inputs:
            urls.append(mm_input.multimodal_url)
            types.append(MMUrlType(mm_input.multimodal_type))
            tensors.append(trans_tensor(mm_input.multimodal_tensor))
            configs.append(trans_config(mm_input.mm_preprocess_config))
    except Exception as e:
        raise Exception(str(e))
    return urls, types, tensors, configs


def trans_output(
    res: MMEmbeddingRes, metadata_only=False, rdma_encoder=None, require_rdma=False
):
    output_pb = MultimodalOutputsPB()
    handles = []
    if res.position_ids is not None and len(res.position_ids) != len(res.embeddings):
        raise ValueError("ViT embedding/position counts do not match")
    try:
        for i, embedding in enumerate(res.embeddings):
            if metadata_only:
                output_pb.multimodal_outputs.add(
                    token_ids=get_multimodal_feature_hash(embedding).tolist()
                )
                continue
            output = output_pb.multimodal_outputs.add()
            descriptor = (
                rdma_encoder.export_embedding(embedding) if rdma_encoder else b""
            )
            if descriptor:
                output.output_rdma.CopyFrom(MMRdmaDescPB.FromString(descriptor))
                handles.append(output.output_rdma.handle)
            else:
                if require_rdma:
                    raise RuntimeError(
                        "Strict ViT RDMA export failed; inline features are disabled"
                    )
                output.multimodal_embedding.CopyFrom(trans_from_tensor(embedding))
            if res.position_ids is not None and res.position_ids[i] is not None:
                output.multimodal_pos_id.CopyFrom(
                    trans_from_tensor(res.position_ids[i])
                )
        return output_pb
    except BaseException:
        if handles:
            rdma_encoder.release(handles)
        raise


class MultimodalRpcServer(MultimodalRpcServiceServicer):
    def __init__(self, mm_process_engine: MMProcessEngine, rdma_encoder=None):
        self.engine = mm_process_engine
        self._status_lock = threading.Lock()
        self._active = 0
        self._status_version = 0
        self.rdma_encoder = rdma_encoder
        config = getattr(mm_process_engine, "vit_config", None)
        self._token_id_cache = MMTokenIdCache(
            getattr(config, "vit_token_cache_item_num", 10000),
            getattr(config, "vit_token_cache_time_window_ms", 30 * 60 * 1000),
        )
        self.require_rdma = getattr(config, "mm_transport_mode", "grpc") == "rdma"
        self.max_requests = (
            getattr(config, "vit_max_concurrent_requests", 32)
            if getattr(mm_process_engine, "_scheduler", None) is not None
            else 1
        )
        if (
            rdma_encoder is None
            and getattr(config, "mm_transport_mode", "grpc") != "grpc"
        ):
            self.rdma_encoder = MMRdmaEncoderOp(config)
            if not self.rdma_encoder.enabled():
                self.rdma_encoder = None

    def ReleaseEmbedding(self, request, context):
        if self.rdma_encoder is not None:
            self.rdma_encoder.release(list(request.handle))
        return EmptyPB()

    def GetWorkerStatus(self, request, context):
        with self._status_lock:
            active = self._active
            # FlexLB ignores version 0. Epoch microseconds also survive restarts.
            self._status_version = max(self._status_version + 1, time.time_ns() // 1000)
            version = self._status_version
        return WorkerStatusPB(
            role="VIT",
            role_type=ROLE_TYPE_VIT,
            alive=True,
            tp_size=1,
            dp_size=1,
            running_query_len=active,
            status_version=version,
            max_seq_len=self.engine.model.model_config.max_seq_len,
        )

    def GetCacheStatus(self, request, context):
        # ViT's feature LRU is not the LLM KV cache advertised to FlexLB.
        return CacheStatusPB()

    def _embed(self, urls, types, tensors, configs, deadline, cancelled):
        res = self.engine.submit(
            urls,
            types,
            tensors=tensors,
            preprocess_configs=configs,
            deadline=deadline,
            cancelled=cancelled,
        )
        if len(res.embeddings) != len(urls):
            raise ValueError("ViT returned an unexpected image count")
        model_config = self.engine.model.model_config
        for embedding in res.embeddings:
            if (
                embedding.dim() != 2
                or embedding.size(0) <= 0
                or embedding.size(1) != model_config.hidden_size
                or embedding.dtype != model_config.compute_dtype
            ):
                raise ValueError("ViT returned an invalid embedding shape or dtype")
        return res

    def _metadata_output(
        self, request, urls, types, tensors, configs, deadline, cancelled
    ):
        self.engine._check_request(deadline, cancelled)
        # Retain only digests: data URLs and tensor payloads can be large.
        keys = [
            hashlib.sha256(item.SerializeToString(deterministic=True)).digest()
            for item in request.multimodal_inputs
        ]
        token_ids = [self._token_id_cache.get(key) for key in keys]
        missing = [i for i, ids in enumerate(token_ids) if ids is None]
        res = MMEmbeddingRes([])
        if missing:
            res = self._embed(
                [urls[i] for i in missing],
                [types[i] for i in missing],
                [tensors[i] for i in missing],
                [configs[i] for i in missing],
                deadline,
                cancelled,
            )
            self.engine._check_request(deadline, cancelled)
            computed = trans_output(res, metadata_only=True)
            self.engine._check_request(deadline, cancelled)
            for i, item in zip(missing, computed.multimodal_outputs):
                # A compact CPU copy owns no embedding or RDMA slot references.
                token_ids[i] = array("i", item.token_ids)
        if (
            sum(len(ids) for ids in token_ids)
            >= self.engine.model.model_config.max_seq_len
        ):
            raise ValueError("ViT output exceeds the model sequence length")
        self.engine._check_request(deadline, cancelled)
        for i in missing:
            self._token_id_cache.put(keys[i], token_ids[i])
        output = MultimodalOutputsPB()
        for ids in token_ids:
            output.multimodal_outputs.add(token_ids=ids)
        return output, res, len(keys) - len(missing), len(missing)

    def _embed_v41(self, request, deadline, cancelled):
        from rtp_llm.models.multimodal.deepseek_v41_processor import image_token_types

        self.engine._check_request(deadline, cancelled)
        if request.metadata_only or request.multimodal_inputs:
            raise ValueError(
                "V4.1 prepared inputs require a feature-only typed request"
            )
        typed = request.v41_inputs
        config = self.engine.model.model_config
        processor = self.engine.model.mm_part.processor_config
        token_types = list(typed.token_types)
        mask = list(typed.image_mask)
        if (
            typed.schema_version != 1
            or len(token_types) != len(mask)
            or len(mask) >= config.max_seq_len
            or mask != [kind != -1 for kind in token_types]
        ):
            raise ValueError("Invalid V4.1 prepared token metadata")
        images = []
        covered = [False] * len(mask)
        previous_end = 0
        for source in typed.images:
            self.engine._check_request(deadline, cancelled)
            height, width = source.n_vit_h, source.n_vit_w
            patch_size = processor.vision_patch_size
            ratio = processor.vision_downsample_ratio
            llm_h, llm_w = (height + ratio - 1) // ratio, (width + ratio - 1) // ratio
            if llm_h * (llm_w + 1) + 2 > processor.vision_max_n_token:
                raise ValueError(
                    "V4.1 prepared image exceeds the per-image token limit"
                )
            shape = [height * width, 3, patch_size, patch_size]
            patches = source.patches
            if (
                height <= 0
                or width <= 0
                or patches.data_type != TensorPB.BF16
                or list(patches.shape) != shape
                or len(patches.bf16_data) != height * width * 3 * patch_size**2 * 2
                or source.processor_identity != processor.identity
            ):
                raise ValueError("Invalid V4.1 prepared image patches or processor")
            types = image_token_types(llm_h, llm_w)
            start, end = source.start, source.start + types.numel()
            if (
                list(source.types) != types.tolist()
                or start < previous_end
                or end > len(mask)
                or token_types[start:end] != types.tolist()
            ):
                raise ValueError("Invalid V4.1 prepared image span")
            covered[start:end] = [True] * types.numel()
            previous_end = end
            images.append(
                dict(
                    start=start,
                    n_vit_h=height,
                    n_vit_w=width,
                    patches=trans_tensor(patches),
                    types=types,
                    content_sha256=source.content_sha256,
                    processor_identity=source.processor_identity,
                )
            )
        if covered != mask:
            raise ValueError("V4.1 image spans do not cover the prepared image mask")
        if sum(mask) * config.hidden_size * 2 >= 1024 * 1024 * 1024 - 1024 * 1024:
            raise ValueError("ViT output exceeds the unary RPC size limit")
        embeddings = []
        for image in images:
            self.engine._check_request(deadline, cancelled)
            result = self.engine.submit_v41([image])
            self.engine._check_request(deadline, cancelled)
            if len(result.embeddings) != 1:
                raise ValueError("V4.1 ViT returned an unexpected image count")
            embedding = result.embeddings[0]
            if (
                tuple(embedding.shape) != (image["types"].numel(), config.hidden_size)
                or embedding.dtype != config.compute_dtype
            ):
                raise ValueError(
                    "V4.1 ViT returned an invalid embedding shape or dtype"
                )
            embeddings.append(embedding)
        return MMEmbeddingRes(
            embeddings, max_batch_size=int(bool(images)), gpu_forwards=len(images)
        )

    def RemoteMultimodalEmbedding(self, multimodal_inputs: MultimodalInputsPB, context):
        cancelled = threading.Event()
        if not context.add_callback(cancelled.set):
            context.abort(grpc.StatusCode.CANCELLED, "ViT request cancelled")
        remaining = context.time_remaining()
        deadline = time.monotonic() + remaining if remaining is not None else None
        if (
            self.require_rdma
            and not multimodal_inputs.metadata_only
            and not multimodal_inputs.support_rdma
        ):
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Strict ViT RDMA requires support_rdma",
            )
        with self._status_lock:
            if self._active >= self.max_requests:
                context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "ViT worker is busy")
            self._active += 1
        output = None
        returned = False
        try:
            is_v41 = multimodal_inputs.HasField("v41_inputs")
            urls, types, tensors, configs = trans_input(multimodal_inputs)
            token_cache_hits = token_cache_misses = 0
            if multimodal_inputs.metadata_only and not is_v41:
                (
                    output,
                    res,
                    token_cache_hits,
                    token_cache_misses,
                ) = self._metadata_output(
                    multimodal_inputs,
                    urls,
                    types,
                    tensors,
                    configs,
                    deadline,
                    cancelled,
                )
            else:
                res = (
                    self._embed_v41(multimodal_inputs, deadline, cancelled)
                    if is_v41
                    else self._embed(urls, types, tensors, configs, deadline, cancelled)
                )
                self.engine._check_request(deadline, cancelled)
                output = trans_output(
                    res,
                    rdma_encoder=(
                        self.rdma_encoder if multimodal_inputs.support_rdma else None
                    ),
                    require_rdma=self.require_rdma,
                )
            self.engine._check_request(deadline, cancelled)
            context.set_trailing_metadata(
                (
                    ("vit-max-batch-images", str(res.max_batch_size)),
                    ("vit-gpu-forwards", str(res.gpu_forwards)),
                )
            )
            logging.info(
                "ViT RPC completed: metadata_only=%s images=%d max_gpu_batch_images=%d "
                "token_cache_hits=%d token_cache_misses=%d",
                multimodal_inputs.metadata_only,
                len(multimodal_inputs.v41_inputs.images) if is_v41 else len(urls),
                res.max_batch_size,
                token_cache_hits,
                token_cache_misses,
            )
            returned = True
            return output
        except futures.CancelledError as error:
            context.abort(grpc.StatusCode.CANCELLED, str(error))
        except TimeoutError as error:
            context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, str(error))
        except ValueError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        except Exception as error:
            logging.exception("ViT request failed")
            context.abort(grpc.StatusCode.INTERNAL, str(error))
        finally:
            # Before returning the response no consumer can have issued a READ.
            # After return, lost responses remain bounded in the pool until restart.
            if output is not None and not returned and self.rdma_encoder is not None:
                self.rdma_encoder.release(
                    [
                        item.output_rdma.handle
                        for item in output.multimodal_outputs
                        if item.HasField("output_rdma")
                    ]
                )
            with self._status_lock:
                self._active -= 1


def _create_rpc_server(service, concurrency):
    service.max_requests = concurrency
    # Embedding admission is capped separately; status/cache/release RPCs need room at saturation.
    rpc_concurrency = concurrency + 2
    executor = futures.ThreadPoolExecutor(max_workers=rpc_concurrency)
    server = grpc.server(
        executor,
        maximum_concurrent_rpcs=rpc_concurrency,
        options=[
            ("grpc.max_send_message_length", 1024 * 1024 * 1024),
            ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
        ],
    )
    add_MultimodalRpcServiceServicer_to_server(service, server)
    return server, executor


def _create_health_server(port, is_ready):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/health":
                self.send_error(404)
                return
            ready = is_ready()
            body = b"ok" if ready else b"unavailable"
            self.send_response(200 if ready else 503)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    return ThreadingHTTPServer(("0.0.0.0", port), Handler)


def _serve_rpc_server(server, engine, executor, shutdown_timeout, health_port=None):
    timeout = ProcessManager.normalize_shutdown_timeout_seconds(shutdown_timeout)
    shutdown_deadline = None
    previous_handlers = {}
    health_server = None
    health_thread = None
    rpc_started = False

    def request_shutdown(signum, frame):
        nonlocal shutdown_deadline
        # Signal handlers must not call gRPC or acquire executor/scheduler locks.
        # Repeated signals must not restart the drain budget.
        if shutdown_deadline is None:
            shutdown_deadline = time.monotonic() + timeout

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_shutdown)
        if health_port is not None:
            health_server = _create_health_server(
                health_port, lambda: rpc_started and shutdown_deadline is None
            )
        server.start()
        rpc_started = True
        if health_server is not None:
            health_thread = threading.Thread(
                target=health_server.serve_forever,
                kwargs={"poll_interval": 0.1},
                name="vit-health",
                daemon=True,
            )
            health_thread.start()
            logging.info("ViT health server listening on port %s", health_port)
        while shutdown_deadline is None:
            if not server.wait_for_termination(timeout=0.1):
                break
    finally:
        rpc_started = False
        graceful = shutdown_deadline is not None
        deadline = shutdown_deadline if graceful else time.monotonic() + timeout
        remaining = max(0.0, deadline - time.monotonic())
        # ThreadPoolExecutor's interpreter-exit hook joins even after
        # shutdown(wait=False). A stuck GPU/handler must not defeat the bound.
        # Terminate the process on expiry; never reclaim published RDMA slots.
        watchdog = threading.Timer(remaining, os._exit, args=(1,))
        watchdog.daemon = True
        watchdog.start()
        try:
            if health_server is not None:
                if health_thread is not None and health_thread.is_alive():
                    health_server.shutdown()
                    health_thread.join()
                health_server.server_close()
            logging.info(
                "Stopping ViT RPC server: drain=%s budget=%.3fs", graceful, remaining
            )
            try:
                # stop() rejects new RPCs immediately and lets existing RPCs drain.
                server.stop(remaining if graceful else 0).wait()
            finally:
                try:
                    engine.stop()
                finally:
                    executor.shutdown(wait=True, cancel_futures=True)
        finally:
            watchdog.cancel()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def vit_start_server(py_env_configs=None):
    if py_env_configs is None:
        py_env_configs = setup_args()
        setup_and_configure_server(py_env_configs)
    if py_env_configs.vit_config.vit_separation != VitSeparation.VIT_SEPARATION_ROLE:
        raise ValueError("The standalone ViT server requires VIT_SEPARATION=1")
    if (
        py_env_configs.parallelism_config.tp_size != 1
        or py_env_configs.parallelism_config.dp_size != 1
    ):
        raise ValueError("The standalone ViT worker requires TP_SIZE=1 and DP_SIZE=1")
    url_data_cache_.resize_cache(py_env_configs.vit_config.url_cache_item_num)
    vit_emb_cache_.resize_cache(py_env_configs.vit_config.mm_cache_item_num)

    # Create and fully initialize engine config (global singleton, ports from config)
    engine_config = EngineConfig.create(py_env_configs, nccl_comm_config=None)
    # ROLE skips create_engine, which normally configures native logging and signals.
    ensure_engine_ops_loaded()
    torch.ops.rtp_llm.init_engine(
        engine_config.profiling_debug_logging_config.ft_alog_conf_path
    )

    # Create model configs (ModelConfig construction is handled in ModelFactory)
    # All model metadata (lora_infos, multi_task_prompt, model_name, template_type, mm_model_config)
    # is set in model_config by create_model_config()
    model_config = ModelFactory.create_model_config(
        model_args=py_env_configs.model_args,
        lora_config=py_env_configs.lora_config,
        kv_cache_config=engine_config.kv_cache_config,
        profiling_debug_logging_config=engine_config.profiling_debug_logging_config,
        generate_env_config=py_env_configs.generate_env_config,
        embedding_config=py_env_configs.embedding_config,
        quantization_config=py_env_configs.quantization_config,
        render_config=py_env_configs.render_config,
    )

    # Update engine_config based on model_config
    ModelFactory.update_engine_config_from_model_config(
        engine_config=engine_config,
        model_config=model_config,
    )

    # Create model using new API
    # All metadata is already in model_config (including mm_model_config)
    # vit_config is needed for multimodal models
    model = ModelFactory.from_model_configs(
        model_config=model_config,
        engine_config=engine_config,
        world_info=get_world_info(
            py_env_configs.server_config,
            py_env_configs.distribute_config,
            py_env_configs.parallelism_config,
        ),
        vit_config=py_env_configs.vit_config,
    )

    engine = MMProcessEngine(model, model.vit_config)
    concurrency = (
        model.vit_config.vit_max_concurrent_requests
        if engine._scheduler is not None
        else 1
    )
    if concurrency <= 0:
        engine.stop()
        raise ValueError("vit_max_concurrent_requests must be positive")
    service = MultimodalRpcServer(engine)
    server, executor = _create_rpc_server(service, concurrency)
    logging.info(f"rpc_server_port: {py_env_configs.server_config.rpc_server_port}")
    server.add_insecure_port(f"0.0.0.0:{py_env_configs.server_config.rpc_server_port}")
    _serve_rpc_server(
        server,
        engine,
        executor,
        py_env_configs.server_config.shutdown_timeout,
        health_port=py_env_configs.server_config.server_port,
    )


if __name__ == "__main__":
    vit_start_server()
