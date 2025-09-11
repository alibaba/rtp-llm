import subprocess

import requests

from rtp_llm.utils.util import stop_server, wait_sever_done

port = 8090
server_process = subprocess.Popen(
    [
        "/opt/conda310/bin/python",
        "-m",
        "rtp_llm.start_server",
        "--checkpoint_path=/mnt/nas1/hf/models--Qwen--Qwen1.5-0.5B-Chat/snapshots/6114e9c18dac0042fa90925f03b046734369472f/",
        "--model_type=qwen_2",
        f"--start_port={port}",
    ]
)
wait_sever_done(server_process, port)

url = f"http://localhost:{port}"
json_data = {
    "prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nhello|im_end|>\n<|im_start|>assistant",
    "generate_config": {"max_new_tokens": 32, "temperature": 0},
}

response = requests.post(url, json=json_data)
print(f"Output 0: {response.json()}")

stop_server(server_process)
