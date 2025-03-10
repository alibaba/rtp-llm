
# 需求：
* 操作系统: Linux
* Python: 3.10
* NVIDIA GPU: Compute Capability 7.0 或者更高 (例如V100, T4, RTX20xx, A100, L4, H100等)

# 进入deploy docker
deploy docker安装了依赖和我们的程序的whl包，进入docker环境，可以直接执行测试。

镜像地址见: [镜像发布历史](./DockerHistory.md)
```bash
git clone https://github.com/alibaba/rtp-llm.git
cd docker
# 注意：以下两条指令，不要使用sudo执行
# CONTAINER_NAME是希望建的容器名称
# 如果是cuda11的环境，IMAGE_NAME为registry.cn-hangzhou.aliyuncs.com/havenask/rtp_llm:{version}_cuda11
# 如果是cuda112的环境，IMAGE_NAME为registry.cn-hangzhou.aliyuncs.com/havenask/rtp_llm:{version}_cuda12
# version见上镜像发布历史
sh ./create_container.sh  <CONTAINER_NAME> <IMAGE_NAME>
sh  CONTAINER_NAME/sshme.sh
```

# 测试
```bash
cd rtp-llm
# 修改test.py中的模型路径，运行一个实际的模型
FT_SERVER_TEST=1 python3 example/test.py
# 也可以启动服务
# start http service
TOKENIZER_PATH=/path/to/tokenizer CHECKPOINT_PATH=/path/to/model MODEL_TYPE=your_model_type FT_SERVER_TEST=1 python3 -m maga_transformer.start_server
# request to server
curl -XPOST http://localhost:8088 -d '{"prompt": "hello, what is your name", "generate_config": {"max_new_tokens": 1000}}'
```
其中 your_model_type 可以在 maga_transformer/models/__init__.py 中查找到