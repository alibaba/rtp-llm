
# 需求：
* 操作系统: Linux
* Python: 3.10
* NVIDIA GPU: Compute Capability 7.0 或者更高 (例如V100, T4, RTX20xx, A100, L4, H100等)

# 进入docker
```
$ git clone https://github.com/alibaba/rtp-llm.git
$ cd docker
$ python3 ./create_container.py create <CONTAINER_NAME> --gpu
$ python3 ./create_container.py enter <CONTAINER_NAME> --gpu
```

# 构建
源码构建使用bazel作为构建系统，推荐版本`5.2.0`。
```
$ cd rtp-llm
$ pip3 install -r ./maga_transformer/requirements_torch_gpu.txt
$ bazel build //maga_transformer:maga_transformer --jobs 100 --verbose_failures
$ # 修改test.py中的模型路径，运行一个实际的模型
$ bazel test //example:test --jobs 100
$ # 单元测试
$ bazel test //maga_transformer/test/model_test/fake_test:all_fake_model_test --jobs 100  --test_output=all
```