from concurrent import futures

import grpc

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MMPreprocessConfigPB,
    MultimodalInputsPB,
    MultimodalOutputPB,
    MultimodalOutputsPB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
    MultimodalRpcServiceServicer,
    add_MultimodalRpcServiceServicer_to_server,
)
from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.distribute.gang_server import GangServer
from rtp_llm.distribute.worker_info import g_worker_info
from rtp_llm.model_factory import ModelFactory
from rtp_llm.ops import MMModelConfig
from rtp_llm.server.server_args.server_args import setup_args
from rtp_llm.utils.grpc_util import trans_from_tensor, trans_tensor
from rtp_llm.utils.mm_process_engine import MMEmbeddingRes, MMProcessEngine
from rtp_llm.utils.multimodal_util import MMUrlType


def trans_config(mm_process_config_pb: MMPreprocessConfigPB):
    return [
        mm_process_config_pb.width,
        mm_process_config_pb.height,
        mm_process_config_pb.min_pixels,
        mm_process_config_pb.max_pixels,
        mm_process_config_pb.fps,
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


def trans_output(res: MMEmbeddingRes):
    output_pb = MultimodalOutputsPB()
    for i in range(len(res.embeddings)):
        output = MultimodalOutputPB(
            multimodal_embedding=trans_from_tensor(res.embeddings[i]),
            multimodal_pos_id=trans_from_tensor(res.position_ids[i]),
        )
        output_pb.multimodal_outputs.append(output)
    return output_pb


class MultimodalRpcServer(MultimodalRpcServiceServicer):
    def __init__(self, mm_process_engine: MMProcessEngine):
        self.engine = mm_process_engine

    def RemoteMultimodalEmbedding(self, multimodal_inputs: MultimodalInputsPB, context):
        urls, types, tensors, configs = trans_input(multimodal_inputs)
        res: MMEmbeddingRes = self.engine.submit(
            urls, types, tensors=tensors, preprocess_configs=configs
        )
        return trans_output(res)


def vit_start_server():
    py_env_configs = setup_args()
    
    # Create GangServer and get gang_info
    gang_server = GangServer(py_env_configs.gang_config, py_env_configs.server_config)
    gang_server.start()
    gang_info = gang_server._gang_info
    
    # Create and fully initialize engine config (global singleton)
    engine_config = EngineConfig.create(py_env_configs, gang_info=gang_info)
    
    # Create model configs (ModelConfig construction is handled in ModelFactory)
    # All model metadata (lora_infos, multi_task_prompt, model_name, template_type)
    # is set in py_model_config by create_model_configs()
    py_model_config, propose_py_model_config = ModelFactory.create_model_configs(
        engine_config=engine_config,
        model_args=py_env_configs.model_args,
        lora_config=py_env_configs.lora_config,
        generate_env_config=py_env_configs.generate_env_config,
        embedding_config=py_env_configs.embedding_config,
    )
    
    mm_model_config = MMModelConfig()
    
    # Create model using new API
    # All metadata is already in py_model_config
    # vit_config is needed for multimodal models
    model = ModelFactory.from_model_configs(
        model_config=py_model_config,
        mm_model_config=mm_model_config,
        engine_config=engine_config,
        gang_info=gang_info,
        vit_config=py_env_configs.vit_config,
        propose_model_config=propose_py_model_config,
    )
    
    # Load default generate config if needed
    if py_env_configs.generate_env_config:
        ModelFactory.load_default_generate_config(model, py_env_configs.generate_env_config)
    
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=200),
        options=[
            ("grpc.max_send_message_length", 1024 * 1024 * 1024),
            ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
        ],
    )
    add_MultimodalRpcServiceServicer_to_server(
        MultimodalRpcServer(MMProcessEngine(model, model.vit_config)), server
    )
    server.add_insecure_port(f"0.0.0.0:{g_worker_info.rpc_server_port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    vit_start_server()
