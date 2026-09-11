import logging
import threading
import time
from concurrent import futures

import grpc

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
    MultimodalOutputPB,
    MultimodalOutputsPB,
    WorkerStatusPB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
    MultimodalRpcServiceServicer,
    add_MultimodalRpcServiceServicer_to_server,
)
from rtp_llm.distribute.distributed_server import get_world_info
from rtp_llm.model_factory import ModelFactory
from rtp_llm.ops import MMRdmaEncoderOp, VitSeparation, get_multimodal_feature_hash
from rtp_llm.server.server_args.server_args import setup_args
from rtp_llm.utils.grpc_util import trans_from_tensor, trans_tensor
from rtp_llm.utils.mm_process_engine import MMEmbeddingRes, MMProcessEngine
from rtp_llm.utils.multimodal_util import MMUrlType, url_data_cache_, vit_emb_cache_

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
        mm_process_config_pb.image_block_start_mod4,
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


def trans_output(res: MMEmbeddingRes, metadata_only=False, rdma_encoder=None):
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

    def RemoteMultimodalEmbedding(self, multimodal_inputs: MultimodalInputsPB, context):
        cancelled = threading.Event()
        if not context.add_callback(cancelled.set):
            context.abort(grpc.StatusCode.CANCELLED, "ViT request cancelled")
        remaining = context.time_remaining()
        deadline = time.monotonic() + remaining if remaining is not None else None
        with self._status_lock:
            if self._active >= self.max_requests:
                context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "ViT worker is busy")
            self._active += 1
        output = None
        returned = False
        try:
            urls, types, tensors, configs = trans_input(multimodal_inputs)
            res: MMEmbeddingRes = self.engine.submit(
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
            self.engine._check_request(deadline, cancelled)
            output = trans_output(
                res,
                multimodal_inputs.metadata_only,
                self.rdma_encoder if multimodal_inputs.support_rdma else None,
            )
            self.engine._check_request(deadline, cancelled)
            context.set_trailing_metadata(
                (
                    ("vit-max-batch-images", str(res.max_batch_size)),
                    ("vit-gpu-forwards", str(res.gpu_forwards)),
                )
            )
            logging.info(
                "ViT RPC completed: metadata_only=%s images=%d max_gpu_batch_images=%d",
                multimodal_inputs.metadata_only,
                len(res.embeddings),
                res.max_batch_size,
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
    server.start()
    try:
        server.wait_for_termination()
    finally:
        server.stop(0).wait()
        engine.stop()
        executor.shutdown(wait=True)


if __name__ == "__main__":
    vit_start_server()
