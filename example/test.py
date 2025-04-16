import os
import json
import asyncio
import logging

from maga_transformer.pipeline import Pipeline
from maga_transformer.model_factory import ModelFactory
from maga_transformer.openai.openai_endpoint import OpenaiEndopoint
from maga_transformer.openai.api_datatype import ChatCompletionRequest, ChatMessage, RoleEnum
from maga_transformer.distribute.worker_info import update_master_info, g_worker_info
from maga_transformer.test.utils.port_util import get_consecutive_free_ports

async def main():
    start_port = get_consecutive_free_ports(1)[0]
    os.environ['START_PORT'] = str(start_port)
    update_master_info('127.0.0.1', start_port)
    g_worker_info.reload()
    os.environ["MODEL_TYPE"] = os.environ.get("MODEL_TYPE", "qwen_2")
    os.environ["CHECKPOINT_PATH"] = os.environ.get("CHECKPOINT_PATH", "Qwen/Qwen2-0.5B-Instruct")
    os.environ["DEVICE_RESERVE_MEMORY_BYTES"] = str(2 * 1024 * 1024 * 1024)
    model_config = ModelFactory.create_normal_model_config()
    model = ModelFactory.from_huggingface(model_config.ckpt_path, model_config=model_config)
    pipeline = Pipeline(model, model.config, model.tokenizer)

    # usual request
    for res in pipeline("<|im_start|>user\nhello, what's your name<|im_end|>\n<|im_start|>assistant\n", max_new_tokens = 100):
        print(res.generate_texts)

    # openai request
    openai_endpoint = OpenaiEndopoint(model.config, model.tokenizer, pipeline.backend_rpc_server_visitor)
    messages = [
        ChatMessage(**{
            "role": RoleEnum.user,
            "content": "你是谁？",
        }),
    ]
    request = ChatCompletionRequest(messages=messages, stream=False)
    response = openai_endpoint.chat_completion(request_id=0, chat_request=request, raw_request=None)
    async for res in response:
        pass
    print((await response.gen_complete_response_once()).model_dump_json(indent=4))

    pipeline.stop()

if __name__ == '__main__':
    asyncio.run(main())
