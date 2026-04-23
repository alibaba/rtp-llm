# P2P Mooncake RDMA 机器执行手册

日期：2026-04-23  
目标分支：`develop/vin/p2p-connector-3`  
目标提交：`develop/vin/p2p-connector-3` 分支最新提交

## 1. 目的

这份手册用于在带 GPU 和 RDMA 设备的机器上，直接复现并验证当前 `P2P Mooncake` 特性的完整测试路径，覆盖：

1. 拉取 `develop/vin/p2p-connector-3` 分支
2. 启动带 GPU、RDMA、模型目录和 Bazel cache 挂载的容器
3. 跑通公共单元测试与集成测试
4. 跑通 Mooncake TCP 参考 smoke
5. 跑通 Mooncake RDMA smoke
6. 读取 remote reuse artifact，确认 smoke 中确实发生了远端数据传输，而不是只把进程拉起

配套一键脚本：`tools/run_p2p_mooncake_rdma_suite.sh`

## 2. 当前分支包含的能力

当前分支已经包含以下内容：

1. Mooncake backend 接入 `IKVCacheSender.h` / `IKVCacheReceiver.h` 现有接口语义，不改上层调用边界
2. `prepare(unique_key)` / `finish(unique_key, status)` 轻量控制面
3. `kMooncake` backend 配置透传与 connector backend 选择逻辑
4. Mooncake adapter 抽象与 Bazel 条件编译
5. Mooncake TCP 模式的真实 smoke 验证
6. RDMA 机器可直接执行的一键测试脚本

当前脚本约定：

1. `cache_store_mooncake_mode=1`
2. Mooncake TCP 参考路径使用 `cache_store_mooncake_transport=tcp`
3. Mooncake RDMA 路径使用 `cache_store_mooncake_transport=rdma`
4. RDMA smoke target 会由脚本基于 TCP smoke target 临时生成，测试结束后默认恢复 `internal_source/rtp_llm/test/smoke/BUILD`

## 3. 前置条件

执行前请确认下面条件全部满足：

1. 机器可访问 GitHub，且对 `alibaba/rtp-llm` 有拉取权限
2. 机器上存在工作区根目录：`/data0/qiongshi.gb/RTP-LLM`
3. 开源仓路径存在：`/data0/qiongshi.gb/RTP-LLM/github-opensource`
4. 内部 smoke 源路径存在：`/data0/qiongshi.gb/RTP-LLM/internal_source`
5. 模型目录已挂载，例如：`/mnt/nas1/hf/Qwen2.5-0.5B-Instruct/config.json`
6. RDMA 设备可见：`/dev/infiniband`、`/sys/class/infiniband`
7. Docker、NVIDIA runtime、`python3`、`git` 可用
8. Bazel 依赖缓存目录存在：`/data0/qiongshi.gb/bazel_deps`
9. 建议预留 Bazel cache 目录：`/data0/qiongshi.gb/.bazel_cache`

建议先在宿主机上执行：

```bash
nvidia-smi
ls -l /dev/infiniband
ls /sys/class/infiniband
rdma link show
ls -l /mnt/nas1/hf/Qwen2.5-0.5B-Instruct/config.json
```

如果 `rdma link show` 为空，说明这台机器还不能验证真实 RDMA one-sided WRITE 数据面。

## 4. 快速开始

### 4.1 拉取目标分支

```bash
cd /data0/qiongshi.gb/RTP-LLM/github-opensource
git fetch origin develop/vin/p2p-connector-3
git checkout develop/vin/p2p-connector-3
git pull --ff-only origin develop/vin/p2p-connector-3
```

### 4.2 一键执行完整验证

```bash
cd /data0/qiongshi.gb/RTP-LLM/github-opensource
bash tools/run_p2p_mooncake_rdma_suite.sh all
```

这条命令会自动完成：

1. 切分支并打印当前提交
2. 删除并重建测试容器 `vin_rtp_rdma_test`
3. 挂载 GPU、RDMA 设备、模型目录、工作区目录和 Bazel cache
4. 跑公共单测和集成测试
5. 跑 Mooncake TCP 参考 smoke
6. 基于 TCP smoke target 临时生成 Mooncake RDMA smoke target
7. 跑 Mooncake RDMA smoke
8. 打印 remote reuse artifact
9. 默认恢复 `internal_source/rtp_llm/test/smoke/BUILD`

## 5. 一键脚本用法

脚本路径：`/data0/qiongshi.gb/RTP-LLM/github-opensource/tools/run_p2p_mooncake_rdma_suite.sh`

帮助命令：

```bash
bash tools/run_p2p_mooncake_rdma_suite.sh --help
```

支持动作：

```bash
bash tools/run_p2p_mooncake_rdma_suite.sh prepare
bash tools/run_p2p_mooncake_rdma_suite.sh test
bash tools/run_p2p_mooncake_rdma_suite.sh smoke-tcp
bash tools/run_p2p_mooncake_rdma_suite.sh smoke-rdma
bash tools/run_p2p_mooncake_rdma_suite.sh all
```

### 5.1 推荐执行方式

第一次在新机器上跑：

```bash
cd /data0/qiongshi.gb/RTP-LLM/github-opensource
bash tools/run_p2p_mooncake_rdma_suite.sh all
```

如果容器已经准备好，只想重跑测试：

```bash
cd /data0/qiongshi.gb/RTP-LLM/github-opensource
bash tools/run_p2p_mooncake_rdma_suite.sh test
```

如果只想跑 TCP 参考 smoke：

```bash
cd /data0/qiongshi.gb/RTP-LLM/github-opensource
RUN_RDMA_SMOKE=0 bash tools/run_p2p_mooncake_rdma_suite.sh test
```

如果只想跑 RDMA smoke：

```bash
cd /data0/qiongshi.gb/RTP-LLM/github-opensource
RUN_TCP_REFERENCE=0 bash tools/run_p2p_mooncake_rdma_suite.sh test
```

如果只想单独执行 smoke：

```bash
cd /data0/qiongshi.gb/RTP-LLM/github-opensource
bash tools/run_p2p_mooncake_rdma_suite.sh smoke-tcp
bash tools/run_p2p_mooncake_rdma_suite.sh smoke-rdma
```

### 5.2 关键环境变量

```bash
BRANCH=develop/vin/p2p-connector-3
WORKSPACE_ROOT=/data0/qiongshi.gb/RTP-LLM
OPEN_SOURCE_REPO=/data0/qiongshi.gb/RTP-LLM/github-opensource
INTERNAL_SOURCE_REPO=/data0/qiongshi.gb/RTP-LLM/internal_source
MODEL_ROOT=/mnt/nas1
BAZEL_DEPS_ROOT=/data0/qiongshi.gb/bazel_deps
BAZEL_CACHE_DIR=/data0/qiongshi.gb/.bazel_cache
CONTAINER_NAME=vin_rtp_rdma_test
IMAGE=hub.docker.alibaba-inc.com/isearch/rtp_llm_dev_gpu_cuda12_9:2025_12_04_19_49_005d702
GPU_CONFIG=sm9x
RUN_TCP_REFERENCE=1
RUN_RDMA_SMOKE=1
RESTORE_INTERNAL_SOURCE=1
```

说明：

1. `RUN_TCP_REFERENCE=1` 表示先跑 Mooncake TCP 参考 smoke，确认控制面和 Mooncake transport 路径整体稳定
2. `RUN_RDMA_SMOKE=1` 表示继续跑 Mooncake RDMA smoke
3. `RESTORE_INTERNAL_SOURCE=1` 表示脚本退出时恢复 `internal_source/rtp_llm/test/smoke/BUILD`
4. 如果你想保留脚本生成的 RDMA smoke target 方便手工调试，可临时设为 `RESTORE_INTERNAL_SOURCE=0`

## 6. 脚本内部会跑哪些测试

### 6.1 公共单元测试 / 集成测试

脚本会先跑：

```bash
//rtp_llm/server/server_args/test:server_args_test
//rtp_llm/cpp/cache/connector/p2p/test:p2p_connector_config_test
//rtp_llm/cpp/cache/connector/p2p/transfer/test:transfer_backend_config_test
//rtp_llm/cpp/cache/connector/p2p/transfer/tcp/test:tcp_sender_receiver_test
//rtp_llm/cpp/cache/connector/p2p/transfer/mooncake:mooncake_backend_stub_test
```

其中 Mooncake backend stub test 会带：

```bash
--define=enable_mooncake_te=true
```

### 6.2 TCP 参考 smoke

脚本会跑：

```bash
//rtp_llm/test/smoke:pd_seperation_prefill_decode_reuse_cache_mooncake_tcp
//rtp_llm/test/smoke:qwen25_05b_base_openai_remote_cache_pd_sep_mooncake_tcp
```

第二条 smoke 是更强的 remote cache case，用于检查 `remote_reuse_len > 0`。

### 6.3 RDMA smoke

脚本会基于 TCP smoke target 临时生成并跑：

```bash
//rtp_llm/test/smoke:pd_seperation_prefill_decode_reuse_cache_mooncake_rdma
//rtp_llm/test/smoke:qwen25_05b_base_openai_remote_cache_pd_sep_mooncake_rdma
```

生成规则：

1. target 名从 `_mooncake_tcp` 改为 `_mooncake_rdma`
2. `cache_store_mooncake_transport tcp` 改为 `cache_store_mooncake_transport rdma`
3. 为避免端口冲突，RPC port 切到 `23645` 到 `23648`

## 7. 期望成功信号

如果脚本执行成功，你应该看到以下结果：

1. `mooncake_backend_stub_test` 通过
2. Mooncake TCP smoke 通过
3. Mooncake RDMA smoke 通过
4. 脚本最后打印 remote reuse artifact，例如：

```json
{
  "reuse_len": 88,
  "local_reuse_len": 0,
  "remote_reuse_len": 88,
  "prefill_remote_reuse_len": 88,
  "decode_remote_reuse_len": 88
}
```

关键判定标准：

1. `remote_reuse_len > 0`
2. `prefill_remote_reuse_len > 0`
3. `decode_remote_reuse_len > 0`

如果这三个值大于 0，说明 smoke 中不仅拉起了服务，而且确实发生了远端数据传输与远端复用。

## 8. 失败时怎么定位

### 8.1 看容器内 RDMA 设备

```bash
docker exec vin_rtp_rdma_test bash -lc 'ls -l /dev/infiniband'
docker exec vin_rtp_rdma_test bash -lc 'rdma link show'
```

### 8.2 看 smoke BUILD 是否已生成 RDMA target

```bash
sed -n '/mooncake_rdma/,+40p' /data0/qiongshi.gb/RTP-LLM/internal_source/rtp_llm/test/smoke/BUILD
```

### 8.3 看 artifact

```bash
python3 - <<'PY'
import json
path = '/data0/qiongshi.gb/RTP-LLM/internal_source/rtp_llm/test/smoke/data/model/qwen25/q_r_l20_remote_cache_pd_sep.query_1.json'
with open(path) as f:
    data = json.load(f)
print(json.dumps(data.get('aux_info', {}), ensure_ascii=False, indent=2))
PY
```

### 8.4 看 Bazel 日志

```bash
find /data0/qiongshi.gb/RTP-LLM/github-opensource/bazel-testlogs -name test.log | tail -n 20
```

如果 smoke 失败，优先检查：

1. 模型目录是否真实挂载到容器里
2. RDMA 设备是否真实透传到容器里
3. `internal_source/rtp_llm/test/smoke/BUILD` 是否成功生成 RDMA target
4. Bazel external repo 是否都能从 `/data0/qiongshi.gb/bazel_deps` 命中

## 9. 手工兜底命令

如果你不想直接跑一键脚本，也可以手工执行。

### 9.1 启动容器

```bash
docker rm -f vin_rtp_rdma_test 2>/dev/null || true

docker run -d --name vin_rtp_rdma_test \
  --privileged \
  --network host \
  --ipc host \
  --gpus all \
  --ulimit memlock=-1 \
  --ulimit nofile=655350:655350 \
  -v /data0/qiongshi.gb:/data0/qiongshi.gb \
  -v /mnt/nas1:/mnt/nas1 \
  -v /dev/infiniband:/dev/infiniband \
  -v /sys/class/infiniband:/sys/class/infiniband \
  -v /sys/class/net:/sys/class/net \
  -v /data0/qiongshi.gb/.bazel_cache:/root/.cache/bazel \
  -w /data0/qiongshi.gb/RTP-LLM/github-opensource \
  hub.docker.alibaba-inc.com/isearch/rtp_llm_dev_gpu_cuda12_9:2025_12_04_19_49_005d702 \
  bash -lc 'sleep infinity'
```

### 9.2 手工跑测试

```bash
docker exec -w /data0/qiongshi.gb/RTP-LLM/github-opensource vin_rtp_rdma_test \
  bazelisk test //rtp_llm/cpp/cache/connector/p2p/transfer/mooncake:mooncake_backend_stub_test \
  --cache_test_results=no \
  --test_output=errors \
  --config=cuda12_9 \
  --config=sm9x \
  --define=enable_mooncake_te=true \
  --override_repository=havenask=/data0/qiongshi.gb/bazel_deps/havenask_3c973500afbd40933eb0a80cfdfb6592274377fb \
  --override_repository=com_google_absl=/data0/qiongshi.gb/bazel_deps/com_google_absl_6f9d96a1f41439ac172ee2ef7ccd8edf0e5d068c \
  --override_repository=cutlass_fa=/data0/qiongshi.gb/bazel_deps/cutlass_fa_bbe579a9e3beb6ea6626d9227ec32d0dae119a49 \
  --override_repository=cutlass=/data0/qiongshi.gb/bazel_deps/cutlass_80243e0b8c644f281e2beb0c20fe78cf7b267061 \
  --override_repository=cutlass_h_moe=/data0/qiongshi.gb/bazel_deps/cutlass_h_moe_19b4c5e065e7e5bbc8082dfc7dbd792bdac850fc \
  --override_repository=cutlass4.0=/data0/qiongshi.gb/bazel_deps/cutlass4_0_dc4817921edda44a549197ff3a9dcf5df0636e7b \
  --override_repository=cutlass3.6=/data0/qiongshi.gb/bazel_deps/cutlass3_6_cc3c29a81a140f7b97045718fb88eb0664c37bd7 \
  --override_repository=rules_cc=/data0/qiongshi.gb/bazel_deps/rules_cc_from_devcache \
  --override_repository=rules_python=/data0/qiongshi.gb/bazel_deps/rules_python_084b877c98b580839ceab2b071b02fc6768f3de6_patched \
  --override_repository=flashinfer_cpp=/data0/qiongshi.gb/bazel_deps/flashinfer_cpp_1c88d650eeec97be3a4dcebe4a9912d7785bc250_patched \
  --override_repository=flash_attention=/data0/qiongshi.gb/bazel_deps/flash_attention_6c9e60de566800538fedad2ad5e6b7b55ca7f0c5_patched \
  --override_repository=rapidjson=/data0/qiongshi.gb/bazel_deps/rapidjson_f54b0e47a08782a6131cc3d60f94d038fa6e0a51_patched \
  --override_repository=grpc=/data0/qiongshi.gb/bazel_deps/grpc_109c570727c3089fef655edcdd0dd02cc5958010_patched
```

## 10. 清理

删除测试容器：

```bash
docker rm -f vin_rtp_rdma_test
```

如果你把 `RESTORE_INTERNAL_SOURCE=0` 设成了关闭，测试结束后记得手工恢复：

```bash
cd /data0/qiongshi.gb/RTP-LLM/internal_source
git restore --source=HEAD -- rtp_llm/test/smoke/BUILD
```

## 11. 与当前 connector 验证结论的关系

当前 `connector` 机器已经做实了：

1. Mooncake TCP 控制面与数据传输路径可用
2. Mooncake TCP smoke 可通过
3. stronger remote cache smoke 中 `remote_reuse_len > 0`

但 `connector` 没有 HCA，不能做真实 RDMA 数据面验证。所以这份 runbook 的目标，就是让你在带 RDMA 的机器上继续把：

1. 真正的 RDMA one-sided WRITE
2. RDMA E2E
3. RDMA smoke

一次性跑全并做实。
