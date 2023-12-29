import os
import asyncio
# from aiostream.stream import list as alist
from typing import Any, Dict, Generator

from maga_transformer.inference import InferenceWorker

async def main() -> None:
    async def _do_inference(worker: InferenceWorker, requests: Any):
        response = worker.inference(**requests)
        await _print_response_in_json(response,)
    
    async def _print_response_in_json(response: Generator[Dict[str, Any], Any, None]):
        for k in range(50):
            result = await response.__anext__()
            print("result = ", result)
    
    os.environ["TOKENIZER_PATH"] = "/mnt/nas1/hf/Qwen-7B"
    os.environ["CHECKPOINT_PATH"] = "/mnt/nas1/hf/Qwen-7B"
    os.environ["MODEL_TYPE"] = "qwen_7b"
    os.environ["ASYNC_MODE"] = "1"

    prompts = [
        "你是谁？"
    ]
    requests = {
        "prompt_batch": prompts,
        "generate_config" : {
            "yield_generator": True,
            "max_new_tokens": 50,
            "top_p": 0,
            "tok_k": 1,
            "temperature": 1.0,
            "calculate_loss": False,
            "num_beams": 1,
        },
        "using_hf_sampling": False,
    }
    
    print("requests = ", requests)
    worker = InferenceWorker()
    await _do_inference(worker, requests)
    worker.stop()
    
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
