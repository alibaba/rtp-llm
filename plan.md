# DSV4 Cache 概念剥离计划

## 目标

把 DSV4 专属的 cache 概念从 `rtp_llm/cpp/cache` 中剥离出去，同时保持 DSV4 的能力完整可用。

`cache` 模块只应该负责通用 cache 机制：

- block pool 的分配和释放
- KV cache group 的通用行为
- GPU prefix/block cache
- memory cache connector
- CP slot mapping
- allocator 调度

DSV4 的特殊行为应该由模型侧生成 cache plan、policy、spec 后注入 cache 层。cache 层不应该直接出现 `DSV4`、`HCA`、`CSA`、`INDEXER` 这类模型语义。

## 目标形态

DSV4 模型侧负责生成一个通用 cache plan，里面描述：

- group type：`FULL`、`SWA`、`LINEAR`
- region identity：通用 `region_id` / `region_name`
- storage kind：paged、fixed、opaque state
- reuse policy：可复用或不可复用
- eviction policy：chain、independent、none
- allocation policy：active tail blocks、显式 block 数量、reserve 计费方式
- 物理 layout：block bytes、entry bytes、entries per block、alignment、dtype

cache 层只消费这些通用配置，不理解 DSV4 的具体语义。

## 迁移步骤

1. 引入通用 cache policy 结构。

   新增通用的 `CacheRegionPolicy` / `CacheGroupSpec`，表达：

   - `region_name`
   - `group_type`
   - `storage_kind`
   - `reuse_policy`
   - `evict_policy`
   - `active_tail_blocks`
   - `explicit_block_num`
   - `reserve_from_paged_budget`

   初期保留现有 DSV4 enum，并做一层兼容映射，降低迁移风险。

2. 将 DSV4 行为从 `SWAKVCacheGroup` 中移出。

   现在 `SWAKVCacheGroup` 会直接检查 `DSV4StateSpec` 和 `HCA_STATE`。应该改成 policy 驱动：

   - `activeTailBlocks()` 读取 `policy.active_tail_blocks`
   - reuse 分配逻辑读取 `policy.reuse_policy`
   - tail block 校验读取通用 policy 标志

   改完后，HCA state 只是一个 SWA-like state group：

   - `active_tail_blocks = 1`
   - `reuse_policy = NON_REUSABLE`

3. 泛化 independent eviction。

   将 DSV4 专属命名改成通用命名：

   - `enable_dsv4_state_block_independent_eviction` 改成 `enable_independent_group_eviction`
   - `state_swa_kv` metrics tag 改成 `evict_policy=independent`
   - DSV4 state/SWA group 列表改成由 policy 提供的 independent eviction group ids

   cache 层只知道某个 group 使用 independent eviction，不需要知道这是 DSV4 state/SWA。

4. 引入通用 opaque cache spec。

   去掉 cache 层对 `DSV4KVSpec` / `DSV4StateSpec` 的依赖，改成通用 `OpaqueKVCacheSpec`，例如：

   ```cpp
   struct OpaqueKVCacheSpec : KVCacheSpec {
       size_t entry_bytes;
       size_t entries_per_block;
       size_t block_size_bytes_alignment;
       DataType store_dtype;
   };
   ```

   DSV4 KV 和 state region 都可以用这个通用物理 layout 表达。

5. 将 DSV4 cache layout 声明移动到模型侧。

   DSV4 不再保留 `rtp_llm/cpp/models/dsv4` 下的专用 builder。模型侧通过
   `KVCacheSpecDesc` 声明 group order、ring/slack、CP slicing 和显式 pool
   block 数，cache 模块只消费通用 desc。

6. 用通用 region identity 替换 DSV4 region enum。

   将 `KVCacheRegionName` 中的 DSV4 专属枚举值替换成通用 region id/name 机制：

   - 内部查找使用 `int region_id`
   - 日志和 metrics 使用 `std::string region_name`
   - 模型侧 builder 注册 `hca_state`、`csa_kv`、`swa_kv` 等名字

   cache 层不再对 DSV4 region name 做 switch 或 if 判断。

7. 清理 cache 目录中残留的 DSV4 命名和测试依赖。

   删除 cache 层直接引用的：

   - `DSV4`
   - `HCA`
   - `CSA`
   - `INDEXER`
   - `HCA_STATE`
   - `skipReuseCacheRegion`

   DSV4 相关测试仍然保留，但模型语义相关的预期应尽量移动到 DSV4 plan 测试中，而不是放在通用 cache 测试里。

## 必须覆盖的测试

- DSV4 cache plan 生成的 group layout 与当前保持一致。
- HCA state 仍然不可复用。
- HCA state 仍然只保留 1 个 active tail block。
- HCA state 仍然使用显式 block 数量，默认 256。
- CSA / indexer / SWA state 的复用行为保持不变。
- independent eviction 对配置的 group 生效。
- prefix tree memory cache 仍能正确处理 DSV4 state/SWA group。
- 现有 DSV4 PD reuse smoke 继续通过。

## 非目标

- 第一阶段不重写 block pool 分配逻辑。
- 第一阶段不改变 DSV4 运行时行为。
- 在所有调用点都改成消费通用 policy 前，不删除兼容映射。

## 最终状态

`rtp_llm/cpp/cache` 中不再包含 DSV4 模型概念。DSV4 通过模型侧生成的 cache plan 配置通用 cache 系统，cache 层只处理 policy 和 opaque physical spec。
