
# 需求：
* 操作系统: Linux
* Python: 3.10
* NVIDIA GPU: Compute Capability 7.0 或者更高 (例如V100, T4, RTX20xx, A100, L4, H100等)

# 进入deploy docker
internal deploy docker安装了依赖和我们的程序的whl包，进入docker环境，可以直接执行测试。
```bash
git clone git@gitlab.alibaba-inc.com:foundation_models/FasterTransformer.git
cd docker
# 注意：以下两条指令，不要使用sudo执行
# 如果是cuda11的环境，IMAGE_NAME为reg.docker.alibaba-inc.com/isearch/rtp_llm_gpu:latest
# 如果是cuda12的环境，IMAGE_NAME为reg.docker.alibaba-inc.com/isearch/rtp_llm_gpu_cuda12:latest
sh ./create_container.sh <DOCKER_NAME> <IMAGE_NAME>
sh  DOCKER_NAME/sshme.sh
```

# 测试
```bash
首先一定要跳出FasterTransformer的路径之外
# 修改test.py中的模型路径，运行一个实际的模型
FT_SERVER_TEST=1 python3 FasterTransformer/example/test.py
# 也可以启动服务
# start http service
TOKENIZER_PATH=/path/to/tokenizer CHECKPOINT_PATH=/path/to/model MODEL_TYPE=your_model_type FT_SERVER_TEST=1 python3 -m rtp_llm.start_server
# request to server
curl -XPOST http://localhost:8088 -d '{"prompt": "hello, what is your name", "generate_config": {"max_new_tokens": 1000}}'
```
其中 your_model_type 可以在 rtp_llm/models/__init__.py 中查找到