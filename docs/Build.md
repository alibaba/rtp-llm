
# 需求：
* 操作系统: Linux
* Python: 3.10
* NVIDIA GPU: Compute Capability 7.0 或者更高 (例如V100, T4, RTX20xx, A100, L4, H100等)

# 进入docker
如果接下来是准备使用whl包来安装，那么进入docker是可选的。如果是从源代码构建，那么进入docker是必选的。
如果本机环境比较复杂，建议进入docker，环境比较干净。
```bash
git clone https://github.com/alibaba/rtp-llm.git
cd docker
# 注意：以下两条指令，不要使用sudo执行
sh ./create_container.sh <CONTAINER_NAME>
sh  CONTAINER_NAME/sshme.sh
```

# 构建
您也可以通过源代码来进行编译。源码构建使用bazel作为构建系统，推荐版本`5.2.0`。
cuda11的环境：
```bash
cd rtp-llm
pip3 install -r ./open_source/deps/requirements_torch_gpu.txt
bazel build //maga_transformer:maga_transformer --jobs 100 --verbose_failures
# 修改test.py中的模型路径，运行一个实际的模型
bazel test //example:test --jobs 100
# 单元测试
bazel test //maga_transformer/test/model_test/fake_test:all_fake_model_test --jobs 100  --test_output=all
```
cuda12的环境：
```bash
cd rtp-llm
pip3 install -r ../open_source/deps/requirements_torch_gpu_cuda12.txt
bazel build //maga_transformer:maga_transformer --jobs 100 --verbose_failures --config=cuda12_2
# 修改test.py中的模型路径，运行一个实际的模型
bazel test //example:test --jobs 100
# 单元测试
bazel test //maga_transformer/test/model_test/fake_test:all_fake_model_test --jobs 100  --test_output=all --config=cuda12_2
```