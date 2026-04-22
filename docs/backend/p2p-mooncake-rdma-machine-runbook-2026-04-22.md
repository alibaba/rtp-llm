# P2P Mooncake RDMA 机器执行手册

日期：2026-04-22
目标分支：`develop/vin/p2p-connector-3`
目标提交：`e1ae1ed7bdcc300732b1769f27e48a603e92945c`

## 1. 用途

这份手册用于在带 RDMA 设备的机器上复现和继续验证当前 P2P Mooncake 特性，覆盖以下内容：

1. 拉取包含 Mooncake TCP transport E2E 和测试报告的目标分支
2. 启动具备 GPU、RDMA 和模型目录挂载的容器
3. 运行 Mooncake TCP transport 的真实数据面 E2E
4. 按需打上 smoke patch，运行 Mooncake TCP mode 的 smoke
5. 为后续切换到真实 RDMA one-sided WRITE 保留一致的工作路径

## 2. 前置条件

执行前请确认：

1. 机器能访问 GitHub，且对 `alibaba/rtp-llm` 有拉取权限
2. 本地工作目录存在：`/data0/qiongshi.gb/RTP-LLM/github-opensource`
3. 模型目录已挂载：`/mnt/nas1`
4. RDMA 设备可见：`/dev/infiniband`、`/sys/class/infiniband`
5. Docker 和 NVIDIA runtime 可用
6. 机器上可用的 Bazel 依赖缓存目录存在：`/data0/qiongshi.gb/bazel_deps`

## 3. 拉取代码

```bash
cd /data0/qiongshi.gb/RTP-LLM/github-opensource
git fetch origin develop/vin/p2p-connector-3
git checkout develop/vin/p2p-connector-3
git rev-parse HEAD
```

期望输出：

```bash
e1ae1ed7bdcc300732b1769f27e48a603e92945c
```

## 4. 启动容器

`docker/create_container.sh` 默认不会挂载 `/data0/qiongshi.gb`、`/mnt/nas1` 和 RDMA 设备，不适合本次验证。建议直接使用下面这条命令启动容器：

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
  -w /data0/qiongshi.gb/RTP-LLM/github-opensource \
  hub.docker.alibaba-inc.com/isearch/rtp_llm_dev_gpu_cuda12_9:2025_12_04_19_49_005d702 \
  bash -lc 'sleep infinity'
```

进入容器：

```bash
docker exec -it vin_rtp_rdma_test bash
```

## 5. 容器内基础准备

```bash
git config --global --add safe.directory /data0/qiongshi.gb/RTP-LLM/github-opensource

cd /data0/qiongshi.gb/RTP-LLM/github-opensource

export BAZEL_OVERRIDES="\
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
--override_repository=grpc=/data0/qiongshi.gb/bazel_deps/grpc_109c570727c3089fef655edcdd0dd02cc5958010_patched"
```

## 6. 先跑 Mooncake TCP transport E2E

这一步的目标是先在 RDMA 机器上复用当前已经在 connector 验证过的 Mooncake TCP 数据面 E2E，确认控制面和 classic TE TCP transport 路径是稳定的。

### 6.1 跑全量 Mooncake backend stub 测试

```bash
bazelisk test //rtp_llm/cpp/cache/connector/p2p/transfer/mooncake:mooncake_backend_stub_test \
  --cache_test_results=no \
  --config=cuda12_9 \
  --config=sm8x \
  $BAZEL_OVERRIDES
```

### 6.2 只跑单 block Mooncake TCP transport E2E

```bash
bazelisk test //rtp_llm/cpp/cache/connector/p2p/transfer/mooncake:mooncake_backend_stub_test \
  --cache_test_results=no \
  --test_arg=--gtest_filter=MooncakeKVCacheClassicTeTest.RealClassicTransferEngineCopiesPayloadOverTcpTransport \
  --config=cuda12_9 \
  --config=sm8x \
  $BAZEL_OVERRIDES
```

### 6.3 只跑多 block Mooncake TCP transport E2E

```bash
bazelisk test //rtp_llm/cpp/cache/connector/p2p/transfer/mooncake:mooncake_backend_stub_test \
  --cache_test_results=no \
  --test_arg=--gtest_filter=MooncakeKVCacheClassicTeTest.RealClassicTransferEngineCopiesMultipleBlocksOverTcpTransport \
  --config=cuda12_9 \
  --config=sm8x \
  $BAZEL_OVERRIDES
```

## 7. Mooncake TCP mode 说明

当前无 RDMA 或先做功能验证时，推荐使用：

- `cache_store_mooncake_mode=1`
- `cache_store_mooncake_transport=tcp`

这组配置的语义是：

1. 控制面继续走 `prepare(unique_key)` / `finish(unique_key, status)` TCP RPC
2. 数据面不走 RDMA，而是走 Mooncake classic TransferEngine 的 TCP transport

它不是最终 RDMA one-sided WRITE 的替代品，但适合先完成：

1. sender/receiver descriptor 协作验证
2. 真正的数据面 copy E2E
3. 多 block / 空洞 block 的真实数据路径验证

## 8. 跑 smoke

### 8.1 先打 smoke patch

仓库里已经准备好 patch 文件：

`docs/backend/p2p-mooncake-tcp-smoke.patch`

如果 RDMA 机器上也具备 `/data0/qiongshi.gb/RTP-LLM/internal_source`，先执行：

```bash
git config --global --add safe.directory /data0/qiongshi.gb/RTP-LLM
cd /data0/qiongshi.gb/RTP-LLM
git apply --check github-opensource/docs/backend/p2p-mooncake-tcp-smoke.patch
git apply github-opensource/docs/backend/p2p-mooncake-tcp-smoke.patch
```

### 8.2 跑 Mooncake TCP mode 的 smoke

```bash
cd /data0/qiongshi.gb/RTP-LLM/github-opensource

bazelisk test //rtp_llm/test/smoke:pd_seperation_prefill_decode_reuse_cache_mooncake_tcp \
  --test_output=errors \
  --config=cuda12_9 \
  --config=sm8x \
  $BAZEL_OVERRIDES
```

### 8.3 smoke 的当前状态说明

这条 smoke target 已经在 connector 上验证过以下事实：

1. target 已成功加入 smoke BUILD
2. Mooncake TCP mode 参数可被 Bazel 正常分析
3. target 能进入大规模 CUDA 编译和 runfiles 准备阶段

connector 上没有把它完整跑完，是因为那台机器在该容器里会触发非常重的首次大规模编译，不是因为 Mooncake TCP mode 参数本身有问题。

## 9. 回滚 smoke patch

如果只想临时验证 smoke，不想保留 patch：

```bash
cd /data0/qiongshi.gb/RTP-LLM/internal_source
git restore --source=HEAD -- rtp_llm/test/smoke/BUILD
```

## 10. 后续 RDMA 验证顺序

当 Mooncake TCP transport 和 smoke 路径确认稳定后，建议继续按下面顺序推进：

1. 切到真实 `cache_store_mooncake_transport=rdma`
2. 验证 one-sided WRITE happy path
3. 验证 timeout / cancel / finish failure
4. 再跑 Mooncake RDMA smoke
5. 最后做性能与稳定性

