# Requirements:
* Operating System: Linux
* Python: 3.10
* NVIDIA GPU: Compute Capability 7.0 or higher (e.g., V100, T4, RTX20xx, A100, L4, H100, etc.)

# Enter deploy docker
The deploy docker contains dependencies and our program's whl package. After entering the docker environment, you can directly execute tests.

Image address see: [Image Release History](./DockerHistory.md)
```bash
git clone https://github.com/alibaba/rtp-llm.git
cd docker
# Note: Do not execute the following two commands with sudo
# CONTAINER_NAME is the desired container name
# For cuda11 environment, IMAGE_NAME is registry.cn-hangzhou.aliyuncs.com/havenask/rtp_llm:{version}_cuda11
# For cuda12 environment, IMAGE_NAME is registry.cn-hangzhou.aliyuncs.com/havenask/rtp_llm:{version}_cuda12
# version see above image release history
sh ./create_container.sh  <CONTAINER_NAME> <IMAGE_NAME>
sh  CONTAINER_NAME/sshme.sh
```

# Testing
```bash
cd rtp-llm
# Modify the model path in test.py and run an actual model
FT_SERVER_TEST=1 python3 example/test.py
# You can also start the service
# start http service
TOKENIZER_PATH=/path/to/tokenizer CHECKPOINT_PATH=/path/to/model MODEL_TYPE=your_model_type FT_SERVER_TEST=1 python3 -m rtp_llm.start_server
# request to server
curl -XPOST http://localhost:8088 -d '{"prompt": "hello, what is your name", "generate_config": {"max_new_tokens": 1000}}'
```
Where your_model_type can be found in rtp_llm/models/__init__.py