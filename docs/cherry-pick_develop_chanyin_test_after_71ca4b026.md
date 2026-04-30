# develop/chanyin/test cherry-pick 迁移记录

## 基本信息

- 当前分支：`develop/chanyin/dsv4_on_dev`
- 待迁移分支：`develop/chanyin/test`
- 基准 commit：`71ca4b0267cd44c10caf02c6335d61b4c5a9cb33`
- 待迁移范围：`71ca4b0267cd44c10caf02c6335d61b4c5a9cb33..develop/chanyin/test`
- 待迁移提交数：7
- 本次动作：按 4 个功能块人工 compact/移植目标分支变更，并保留最终冲突处理记录
- 工作区状态：分析前已有未跟踪项 `.codex`、`lot_update_cb ctor param + invocation.`、`t:cuda13`，本次未处理这些文件。

## 执行过程记录

1. 读取 `using-superpowers` skill，并按要求先做分析，不直接修改代码或执行 cherry-pick。
2. 确认当前分支和目标分支：
   - `git branch --show-current`
   - `git rev-parse --verify develop/chanyin/test`
3. 确认基准 commit 存在：
   - `git cat-file -t 71ca4b0267cd44c10caf02c6335d61b4c5a9cb33`
4. 列出待迁移提交：
   - `git log --reverse --oneline 71ca4b0267cd44c10caf02c6335d61b4c5a9cb33..develop/chanyin/test`
5. 统计待迁移范围变更：
   - `git diff --stat 71ca4b0267cd44c10caf02c6335d61b4c5a9cb33..develop/chanyin/test`
   - 结果为 69 个文件，约 3546 行新增、1536 行删除。
6. 用 `git merge-tree --write-tree --merge-base=<base> HEAD develop/chanyin/test` 预估整段变更合入当前分支的文本冲突。
7. 对 7 个原始提交分别用 `merge-tree` 估算独立应用到当前 HEAD 时的冲突面。注意：这不是连续 cherry-pick 的精确结果，只用于判断功能块的高风险文件。
8. 用户确认最终方案后，按以下原则执行移植：
   - 最终命名采用 `region-name`，将 `KVCacheAttnType` / `layer_attn_to_group_id` / `group_attn_types` 收敛为 `KVCacheRegionName` / `layer_region_to_group_id` / `group_region_names`。
   - Allocator 架构以当前分支的 `HybridKVCacheAllocator` 为公共基类，保留 `HybridTypeKVCacheAllocator` 与 `HybridPoolKVCacheAllocator` 两个派生版本。
   - DSV4 迁移到 7 个 `KVCacheGroup`：3 个 `FullKVCacheGroup` + 4 个 `SWAKVCacheGroup`，不再使用 `LinearKVCacheGroup` 表达 DSV4 的 tail-window region。
9. 处理 pybind / Python stub / WriteCacheStoreOp 调用链：
   - `write_cache_store` 不再暴露 `group_id` / `kv_block_stride_bytes` 参数，由 C++ 侧按 layer + region/group 信息规划。
   - `deepseek_v4_model.py` 的写 cache-store 调用同步更新。
10. 处理 Bazel 依赖：
    - `rtp_llm/cpp/cache:kv_cache_transfer_planner`
    - `rtp_llm/cpp/cache:batch_kv_cache_resource`
    - `rtp_llm/cpp/cache:cache_group_type`
    - cuda13 下 Torch CUDA12 wheel 运行库链接依赖：`libcudart.so.12`、`libcublas.so.12`、`libcublasLt.so.12`、`libcufft.so.11`、`libcupti.so.12`。
11. 修正 DSV4 配置测试，使其匹配当前实现的 BF16-only DSV4 KV entry 布局：
    - `KV_ENTRY_BYTES = 1024`
    - `INDEXER_ENTRY_BYTES = 256`
    - DSV4 group type 断言为 3 个 `FULL` + 4 个 `SWA`。

## 原始提交列表

```text
ff48c7de3 feat: support PD separation KV cache transfer by layer region name
40fa748db feat: support MemoryConnector KV block access by layer region name
0b930f76b fix: refine HybridPolKVCacheAllocator
937e3d877 fix: HyrbidPoolKVCacheAllocator
7413c4ae7 fix: hybrid kvcache allocator rollback
23fae482c feat: add SWAKVCacheGroup for sliding window attention
a6f9798ad fix: cr problems
```

## 建议 compact 后的功能提交

### 1. KV cache region-name 语义迁移

建议包含：
- `ff48c7de3`
- `40fa748db`
- `a6f9798ad` 中与 region-name、remote connector、block pool helper 相关的小修

功能边界：
- 将原先 `KVCacheAttnType` / `layer_attn_to_group_id` / `layer_attn_types` 语义迁移为 `KVCacheRegionName` / `layer_region_to_group_id` / `layer_group_types` 与 `group_region_names`。
- PD separation cache transfer 按 layer + region_name 生成 cache key。
- MemoryConnector 按 layer + region_name 访问和复制 KV block。
- Linear group 传输最后两个 block，而不是只传一个。

主要风险：
- 当前分支 DSV4 仍大量使用 `layer_attn_*` 命名和 `KVCacheAttnType` 语义。
- 如果直接套用目标分支命名，会影响 C++ cache、pybind、Python stub 和 DSV4 配置生成链路。

### 2. HybridPool/HybridType allocator 独立 pool 与 block 分配逻辑

建议包含：
- `0b930f76b`
- `937e3d877`
- `7413c4ae7`
- `a6f9798ad` 中与 allocator、BlockPoolConfigHelper、pool ratio 相关的小修

功能边界：
- 引入或调整独立 block pool 的显存比例分配。
- 调整 `HybridPoolKVCacheAllocator`、`HybridTypeKVCacheAllocator`、`CacheConfigCreator`、`BlockPoolConfigHelper` 的分组 block 数和 layout 计算。
- 增加相关单测。

主要风险：
- 当前分支已有 DSV4 7-pool KV cache 基础设施，并且曾把 `HybridKVCacheAllocator` 重命名/收敛到 `HybridTypeKVCacheAllocator`。
- 目标分支会重新引入 `HybridKVCacheAllocator.{cc,h}`，这在文本上可能不冲突，但在设计上会与当前分支 allocator 命名和职责边界冲突。

### 3. Decode cache load planner 与 region 工具

建议包含：
- `937e3d877`
- `7413c4ae7`
- `23fae482c` 中的 planner/test 相关变更

功能边界：
- 新增 `KVCacheLayerRegionUtils.{cc,h}`。
- 新增 `KVCacheTransferPlanner.{cc,h}`。
- 新增 `DecodeCacheLoadPlanner.{cc,h}` 和测试。
- 从 `DecodeRpcServer` 中拆出或简化 cache load 规划逻辑。

主要风险：
- `DecodeRpcServer.cc` 是最高风险冲突文件之一，当前分支和目标分支都改过 PD separation、cache load 和 DSV4 相关路径。
- 若先合 allocator 再合 planner，`DecodeRpcServer` 冲突会更集中；若先合 planner，再合 region-name，可能出现重复调整。

### 4. SWA KV cache group 支持

建议包含：
- `23fae482c`
- `a6f9798ad` 中与 SWA group 和 remote connector 相关的小修

功能边界：
- 新增 `SWAKVCacheGroup.{cc,h}`。
- `CacheGroupType` 增加 SWA 相关类型。
- `HybridConfigCreator` 支持 sliding-window attention group。
- 增加 `SWAKVCacheGroupTest`、`KVCacheTransferPlannerTest` 等测试。

主要风险：
- 当前分支的 DSV4 cache group 已有 7 组、多 region、多 allocator 逻辑，SWA group 应当按当前分支的 `layer_attn_*` 或目标分支的 `layer_region_*` 最终决策来接入，不能机械套用。

## 整段变更预估文本冲突

用以下命令按基准 commit 作为 merge-base 预估：

```bash
git merge-tree --write-tree --merge-base=71ca4b0267cd44c10caf02c6335d61b4c5a9cb33 --name-only HEAD develop/chanyin/test
```

预估冲突文件：

```text
rtp_llm/cpp/cache/BlockPoolConfigHelper.h
rtp_llm/cpp/cache/CacheConfig.h
rtp_llm/cpp/cache/CacheConfigCreator.cc
rtp_llm/cpp/cache/CacheGroupType.h
rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.cc
rtp_llm/cpp/model_rpc/DecodeRpcServer.cc
rtp_llm/cpp/models/PyWrappedModel.h
rtp_llm/models_py/bindings/OpDefs.cc
rtp_llm/models_py/bindings/OpDefs.h
rtp_llm/models_py/bindings/common/WriteCacheStoreOp.cc
rtp_llm/models_py/bindings/core/ExecOps.cc
rtp_llm/models_py/bindings/core/test/ExecOpsTest.cc
```

这些是文本冲突。另有设计冲突需要手工评估：

- `HybridKVCacheAllocator.{cc,h}`：目标分支新增，但当前分支没有该文件，当前分支历史中已经有 `refactor(cache): rename HybridKVCacheAllocator to HybridTypeKVCacheAllocator`。
- `KVCacheTransferPlanner.{cc,h}`、`DecodeCacheLoadPlanner.{cc,h}`：目标分支新增，当前分支没有；是否保留为独立模块要结合当前 `DecodeRpcServer` 和 DSV4 decode path。
- `rtp_llm/models_py/modules/dsv4/test/BUILD`：整段 merge-tree 里显示目标侧删除，当前侧也存在 DSV4 测试布局变更，需要确认不要误删当前分支 DSV4 测试。

## 冲突原因分组

### A. 命名和数据结构语义冲突

涉及文件：

```text
rtp_llm/cpp/cache/CacheConfig.h
rtp_llm/cpp/cache/BufferTypes.h
rtp_llm/cpp/cache/BatchKVCacheResource.h
rtp_llm/cpp/cache/KVCacheResource.{cc,h}
rtp_llm/cpp/models/PyWrappedModel.{cc,h}
rtp_llm/models_py/bindings/OpDefs.{cc,h}
rtp_llm/ops/librtp_compute_ops/__init__.pyi
```

当前分支仍以 `KVCacheAttnType`、`layer_attn_to_group_id`、`layer_attn_types` 为主；目标分支改成 region-name 表达。这里应先定最终 API 名称，再处理其它冲突。

### B. Allocator 架构冲突

涉及文件：

```text
rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.{cc,h}
rtp_llm/cpp/cache/HybridTypeKVCacheAllocator.{cc,h}
rtp_llm/cpp/cache/HybridKVCacheAllocator.{cc,h}
rtp_llm/cpp/cache/KVCacheAllocator.{cc,h}
rtp_llm/cpp/cache/BlockPoolConfigHelper.h
rtp_llm/cpp/cache/CacheConfigCreator.cc
```

当前分支已有 DSV4 7-pool 逻辑，目标分支又引入独立 pool、pool ratio 和 HybridKVCacheAllocator 相关修正。建议不要逐行取一边，而是以当前分支 allocator 架构为基底，把目标分支的能力移植进去。

### C. PD separation / decode RPC 冲突

涉及文件：

```text
rtp_llm/cpp/model_rpc/DecodeRpcServer.cc
rtp_llm/cpp/model_rpc/DecodeCacheLoadPlanner.{cc,h}
rtp_llm/cpp/cache/KVCacheTransferPlanner.{cc,h}
rtp_llm/cpp/utils/KVCacheUtils.h
rtp_llm/models_py/bindings/common/WriteCacheStoreOp.cc
rtp_llm/models_py/bindings/core/ExecOps.cc
```

这里同时涉及 cache key 生成、按 region 写 cache store、linear group block 选择、decode load planning。建议在 region-name API 决策完成后再处理。

### D. 测试和 BUILD 文件冲突

涉及文件：

```text
rtp_llm/cpp/cache/BUILD
rtp_llm/cpp/cache/test/BUILD
rtp_llm/cpp/cache/test/HybridTypeKVCacheAllocatorTest.cc
rtp_llm/cpp/cache/test/SWAKVCacheGroupTest.cc
rtp_llm/cpp/cache/test/KVCacheTransferPlannerTest.cc
rtp_llm/cpp/model_rpc/BUILD
rtp_llm/cpp/model_rpc/test/BUILD
rtp_llm/cpp/model_rpc/test/DecodeCacheLoadPlannerTest.cc
rtp_llm/models_py/bindings/core/test/ExecOpsTest.cc
```

测试文件冲突主要来自当前分支已重构 DSV4/cache 测试布局，目标分支新增了 planner、SWA 和 region-name 写入测试。

## 建议 cherry-pick 策略

1. 先不要逐个原始 commit 直接 cherry-pick。原始提交里多次修正同一组文件，直接 cherry-pick 会反复解决同一类冲突。
2. 先人工形成 4 个 compact commit：
   - region-name API 和 cache transfer
   - allocator 独立 pool / block 分配
   - decode cache load planner
   - SWA KV cache group
3. 第一优先决策：当前分支是否接受 `KVCacheRegionName` 命名替换 `KVCacheAttnType`。如果接受，应先做这个底座 commit；如果不接受，则目标分支所有 region-name 逻辑都需要映射回当前 `layer_attn_*` API。
4. 第二优先决策：是否重新引入 `HybridKVCacheAllocator`。从当前分支历史看，更合理的方向可能是把目标逻辑 port 到 `HybridTypeKVCacheAllocator` / `HybridPoolKVCacheAllocator`，避免恢复旧类名。
5. 最后处理测试和 BUILD，把新增测试按当前分支测试布局接入，而不是照搬目标分支删除/新增。

## 推荐验证

完成实际移植后建议至少跑：

```bash
bazel test //rtp_llm/cpp/cache/test:all
bazel test //rtp_llm/cpp/model_rpc/test:all
bazel test //rtp_llm/models_py/bindings/core/test:all
```

如果最终启用 DSV4 相关路径，还需要补跑当前分支已有的 DSV4 KV cache、decode 和 precision 测试。

## 实际验证记录

### 已通过

```bash
docker exec -u liudu.ld chanyin_dev_cd13 bash -lc \
  'cd /home/liudu.ld/RTP-LLM/github-opensource && \
   bazelisk --batch build //rtp_llm/cpp/cache:kv_cache_allocator \
     --config=cuda13 --verbose_failures --noshow_progress --curses=no'
```

结果：通过。

```bash
docker exec -u liudu.ld chanyin_dev_cd13 bash -lc \
  'cd /home/liudu.ld/RTP-LLM/github-opensource && \
   bazelisk --batch build //rtp_llm/cpp/cache/test:dsv4_cache_test \
     --config=cuda13 --verbose_failures --noshow_progress --curses=no'
```

结果：通过。

### 用户指定完整测试

执行命令：

```bash
docker exec -it -u liudu.ld chanyin_dev_cd13 bash -lc \
  'cd /home/liudu.ld/RTP-LLM/github-opensource && \
   bazelisk --batch test //rtp_llm/cpp/cache/test:all \
     --config=cuda13 --verbose_failures --cache_test_results=no \
     --test_env='"'"'CUDA_VISIBLE_DEVICES=4,5'"'"''
```

结果：通过。

```text
Executed 10 out of 10 tests: 10 tests pass.
```

补充修正：

- `LinearKVCacheGroup` ring/fixed-cap 模式的 `match()` 设计为 veto prefix reuse，避免恢复到错误的 sliding-window/state 位置；同步更新 `RingMatchVetoesPrefixReuse` 测试断言。
- DSV4 的 7 个 group 中只有 0/1/2 这 3 个 paged `FULL` group 参与 prefix-cache restore；3/4/5/6 这 4 个 `SWA` tail/state group 保留 cache entry，但不直接 prefix restore。同步更新 DSV4 prefix-cache 测试断言。
