import asyncio
import os

import rtp_llm.models
from rtp_llm.config.py_config_modules import StaticConfig
from rtp_llm.distribute.worker_info import g_worker_info, update_master_info
from rtp_llm.frontend.frontend_worker import FrontendWorker
from rtp_llm.model_factory import ModelFactory
from rtp_llm.openai.api_datatype import ChatCompletionRequest, ChatMessage, RoleEnum
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.test.utils.port_util import PortsContext


async def main():
    with PortsContext(None, 1) as ports:
        start_port = ports[0]
        StaticConfig.server_config.start_port = start_port
        update_master_info("127.0.0.1", start_port)
        g_worker_info.reload()
        StaticConfig.model_config.model_type = "qwen_2"
        StaticConfig.model_config.checkpoint_path = "Qwen/Qwen2-0.5B-Instruct"
        os.environ["DEVICE_RESERVE_MEMORY_BYTES"] = str(3 * 1024 * 1024 * 1024)
        model_config = ModelFactory.create_normal_model_config()
        model = ModelFactory.from_huggingface(
            model_config.ckpt_path, model_config=model_config
        )
        backend_rpc_server_visitor = BackendRPCServerVisitor(model.config, False)
        pipeline = FrontendWorker(
            model.config, model.tokenizer, backend_rpc_server_visitor
        )

        # usual request
        for res in pipeline(
            "<|im_start|>user\nhello, what's your name<|im_end|>\n<|im_start|>assistant\n",
            max_new_tokens=100,
        ):
            print(res.generate_texts)

        # openai request
        openai_endpoint = OpenaiEndpoint(
            model.config, model.tokenizer, pipeline.backend_rpc_server_visitor
        )
        messages = [
            ChatMessage(
                **{
                    "role": RoleEnum.user,
                    "content": "你是谁？",
                }
            ),
        ]
        request = ChatCompletionRequest(messages=messages, stream=False)
        response = openai_endpoint.chat_completion(
            request_id=0, chat_request=request, raw_request=None
        )
        async for res in response:
            pass
        print((await response.gen_complete_response_once()).model_dump_json(indent=4))

        model.stop()


if __name__ == "__main__":
    asyncio.run(main())
