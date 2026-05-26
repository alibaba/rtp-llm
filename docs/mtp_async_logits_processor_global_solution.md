# MTP Async Logits Processor Global Solution

Date: 2026-05-25

Related review doc: `docs/mtp_async_logits_processor_cross_review.md`

## Goal

Make `mtp + async` logits processor behavior correct by construction, instead of relying on scattered sync points and processor-specific fixes.

The target design introduces a shared protocol for:

1. request constraint normalization,
2. logits processor capability declaration,
3. MTP spec verify artifact ownership,
4. token commit and processor state versioning,
5. explicit fallback/reject decisions when a processor cannot be safely handled.

## Non-Goals

- Do not add another one-off grammar-only path.
- Do not silently skip unsupported stateful processors.
- Do not use `isStateful()` as a catch-all for commit update, score-batch filtering, and length validation.
- Do not rely on CUDA event readiness as proof that a reusable tensor has already been consumed.

## 现状问题与目标改造

这一节只回答三个问题：

1. 当前代码链路怎么走；
2. 问题具体发生在哪个文件和函数；
3. 最终需要改成什么链路。

### 现状链路：当前代码怎么走

```mermaid
flowchart TD
  A["MtpExecutor::decodeStep<br/>执行 draftModelDecode"] --> B["启动 spec logits worker<br/>MtpExecutor.cc:958-990"]
  A --> C["执行 target verify forward<br/>MtpExecutor.cc:998-1001"]
  B --> D["SpecLogitsVerifyRunner::buildInline<br/>生成 mask/cap"]
  D --> E["LaunchResult:<br/>spec_vocab_mask_gpu<br/>spec_cap_gpu<br/>ready_event<br/>has_active_processor"]
  C --> F["采样前等待 spec worker 结束<br/>MtpExecutor.cc:1040-1042"]
  E --> G["gatherSpecSamplerInput 复制结果<br/>MtpBatchStreamProcessor.cc:343-347"]
  G --> H["LogitsProcessorStates::batchProcess<br/>等待 ready_event 并 apply mask"]
  H --> I["跳过所有 SpecLogitsProcessor<br/>LogitsProcessorStates.cc:23-25"]
  I --> J["sampler + rejection + cap"]
  J --> K["async bookkeeping/specUpdate<br/>稍后更新 stream host state"]

  D --> P1["P1: 没有 applied_processor_ids<br/>只有 has_active_processor"]
  I --> P2["P2: skip 粒度太粗<br/>直接跳过所有 Spec processor"]
  E --> P3["P3: ready_event 只表示生产完成<br/>没有 consumer-done event"]
  K --> P4["P4: 下一轮可能读到旧的<br/>processor state，缺少 version"]

  classDef problem fill:#ffebee,stroke:#c62828,color:#7f0000
  class P1,P2,P3,P4 problem
```

### 问题定位：当前错在哪里

| 问题 | 当前位置 | 为什么错 | 全局改法 |
|---|---|---|---|
| Spec mask 只能表达“存在某个 spec processor” | `SpecLogitsVerifyRunner::LaunchResult::has_active_processor`，在 `SpecLogitsVerifyRunner.cc:176-179` 设置 | sampler 不知道到底哪个 processor 被真正应用。如果 processor 不 eligible 或者实际是 allow-all，后续仍可能被跳过。 | 改成 `SpecVerifyArtifact.applied_processor_ids`。只有 processor id 在集合里，sampler 才能跳过它。 |
| sampler 看到 spec mask 后跳过所有 spec processor | `LogitsProcessorStates.cc:23-25` | 判断条件是 processor 类型，不是 artifact 覆盖范围，会丢 Grammar/ThinkMode 等约束。 | `batchProcess()` 改成逐 processor 检查 `artifact.applied_processor_ids.contains(processor_id)`。 |
| artifact 只有 producer-ready event，没有 consumer-done event | 当前 `LaunchResult` 只有 `ready_event`，没有 consumed event | `ready_event` 只保护 H2D 完成，不能证明 `masked_fill_` 和 cap 逻辑已经读完 tensor。 | 新增 `artifact_consumed_event`；artifact buffer 只能在该 event 后复用。 |
| async commit 晚于下一轮调度 | `MtpExecutor.cc:2035-2041` 已经说明下一轮可能早于 worker-side `specUpdate` 提交 host state | Grammar/ThinkMode 可能基于旧状态生成下一轮 spec mask。 | 增加 processor state epoch：spec verify 读取版本 `V`，commit 发布 `V+1`，下一轮只有读 `V+1` 时才等待。 |
| unsupported / ineligible processor 没有明确策略 | `SpecLogitsVerifyRunner.cc:127-130` 遇到 ineligible processor 直接 `return {}` | 空结果同时可能表示无 processor、不支持、错误，语义混在一起。 | 增加 `ProcessorPlan` 和 `SpecVerifyStatus`: `Applied`, `Noop`, `Unavailable`, `Error`；由 policy 决定 reject / disable MTP / disable async。 |

## 目标链路：最终需要怎么改

最终目标不是把 async 关掉，而是保留 CPU/GPU overlap，同时把每个异步交接点都变成明确的 event 依赖，并把“跳过 processor”的依据改成 artifact 里的 `applied_processor_ids`。

```mermaid
flowchart TD
  A["1. stream 初始化<br/>构建 ProcessorPlan"] --> B{"所有 processor 都支持<br/>MTP async 吗?"}
  B -->|no| B1["reject / disable_mtp / disable_async"]
  B -->|yes| C["2. decode round 读取<br/>processor state version V"]

  C --> D["3A. CPU spec worker<br/>构建 SpecVerifyArtifact"]
  C --> E["3B. GPU target stream<br/>执行 target verify forward"]

  D --> D1["artifact 包含:<br/>vocab_mask_gpu<br/>cap_gpu<br/>applied_processor_ids"]
  D1 --> D2["record artifact_ready_event"]
  E --> E1["record target_logits_ready_event<br/>或使用同 stream 顺序"]

  D2 --> F["4. Sampler 只等待<br/>自己要读的数据"]
  E1 --> F
  F --> G["apply artifact mask<br/>只跳过 applied_processor_ids"]
  G --> H["sampling + rejection + cap"]
  H --> I["record accepted_tokens_ready_event"]
  G --> J["record artifact_consumed_event"]

  I --> K["5. Bookkeeping worker 等待<br/>accepted_tokens_ready_event"]
  K --> L["commit stream tokens"]
  L --> M["commit 所有<br/>needs_commit_update processors"]
  M --> N["发布 processor state version V+1"]
  N --> O["record commit_done_event"]

  J --> P["artifact buffer pool 等待<br/>artifact_consumed_event 后复用"]
  O --> Q{"下一轮需要读取<br/>processor state V+1?"}
  Q -->|yes| R["wait commit_done_event"]
  Q -->|no| S["不做 broad sync，继续调度"]

  classDef target fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
  classDef wait fill:#fff8e1,stroke:#f9a825,color:#5d4300
  classDef fallback fill:#fff3e0,stroke:#ef6c00,color:#4e2600
  class A,C,D,E,D1,G,H,L,M,N,S target
  class F,K,P,Q,R wait
  class B1 fallback
```

### Event 等待总图

这一张图只表达 event 关系：谁发送 event，谁同步等待，等待保护哪份数据。

```mermaid
flowchart LR
  R1["发送方: spec logits worker<br/>mask/cap H2D 入队后"] --> E1(("artifact_ready_event")) --> W1["等待方: sampler<br/>apply spec mask 前"] --> D1["保护数据: vocab_mask_gpu<br/>cap_gpu<br/>applied_processor_ids"]

  R2["发送方: target verify stream<br/>logits 写入后"] --> E2(("target_logits_ready_event<br/>或同 stream 顺序")) --> W2["等待方: sampler<br/>sampling 前"] --> D2["保护数据: target verify logits"]

  R3["发送方: sampler stream<br/>accepted tokens 写入后"] --> E3(("accepted_tokens_ready_event")) --> W3["等待方: bookkeeping worker<br/>CPU commit 前"] --> D3["保护数据: accept_len<br/>accept_tokens<br/>next device state"]

  R4["发送方: sampler stream<br/>mask/cap 读操作入队后"] --> E4(("artifact_consumed_event")) --> W4["等待方: artifact buffer pool<br/>复用前"] --> D4["保护数据: artifact GPU buffers<br/>pinned CPU staging buffers"]

  R5["发送方: bookkeeping worker<br/>token + processor commit 后"] --> E5(("commit_done_event")) --> W5["等待方: 下一轮 decode<br/>读取 processor state V+1 前"] --> D5["保护数据: Grammar / ThinkMode /<br/>其他 needs_commit_update state"]
```

### 最终代码改造清单

| 改造区域 | 最终改法 |
|---|---|
| `SpecLogitsVerifyRunner::LaunchResult` | 替换为 `SpecVerifyArtifact`: `vocab_mask_gpu`, `cap_gpu`, `applied_processor_ids`, `artifact_ready_event`, `artifact_consumed_event`, owned/pool buffer handle。 |
| `SpecLogitsVerifyRunner::buildInline()` | 每个 processor 返回明确 `SpecVerifyStatus`。只有 processor 真实贡献了 mask/cap 语义时，才把 id 放入 `applied_processor_ids`。 |
| `MtpBatchStreamProcessor::gatherSpecSamplerInput()` | 传完整 artifact 给 sampler inputs，不再只传 `has_active_processor` 和 mask tensor。 |
| `LogitsProcessorStates::batchProcess()` | 把“跳过所有 `SpecLogitsProcessor`”改成“只有 `processor_id` 在 `applied_processor_ids` 内才跳过”；否则走正常 `process()`，或者 fallback/reject。 |
| `MtpExecutor::decodeStep()` | 保留 target forward 和 spec worker overlap，但不再把 spec worker 完成当成语义正确。sampler 等 `artifact_ready_event`；commit worker 等 `accepted_tokens_ready_event`；下一轮只有读 processor state 时等 `commit_done_event`。 |
| `GenerateStream` / processor state | 增加 processor state epoch。先 commit accepted tokens 和所有 `needs_commit_update` processor，再发布 version `V+1`。 |

## Core Invariants

### Invariant 1: Spec artifact ownership

`SpecVerifyArtifact` owns or leases the tensors it returns. The runner must not reuse any backing storage until the consumer stream records completion.

Required fields:

```cpp
struct SpecVerifyArtifact {
    torch::Tensor vocab_mask_gpu;
    torch::Tensor cap_gpu;
    std::vector<ProcessorId> applied_processor_ids;
    std::shared_ptr<torch::Event> artifact_ready_event;
    std::shared_ptr<torch::Event> artifact_consumed_event;
};
```

Rules:

- producer records `artifact_ready_event` after mask/cap H2D copies are queued;
- sampler/cap path waits on `artifact_ready_event`;
- after `masked_fill_` and cap application are queued, sampler stream records `artifact_consumed_event`;
- buffer pool can reuse artifact storage only after waiting on `artifact_consumed_event`;
- pinned CPU staging buffers follow the same lifetime rule.

Lifecycle:

```mermaid
flowchart LR
  A["Spec worker fills<br/>mask/cap buffers"] --> B["record artifact_ready_event"]
  B --> C["Sampler waits<br/>artifact_ready_event"]
  C --> D["Sampler reads<br/>vocab_mask_gpu + cap_gpu"]
  D --> E["Sampler stream records<br/>artifact_consumed_event"]
  E --> F["Buffer pool may<br/>reuse storage"]

  B -.-> X["Do not reuse here:<br/>artifact_ready_event is only producer-done"]

  classDef safe fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
  classDef bad fill:#ffebee,stroke:#c62828,color:#7f0000
  class A,B,C,D,E,F safe
  class X bad
```

### Invariant 2: Applied processor semantics

Sampler preprocessing can skip a processor only if that exact processor was applied into the spec verify artifact.

Wrong semantic:

```text
has_active_processor == true
```

Correct semantic:

```text
applied_processor_ids contains processor.id()
```

Why this matters:

```mermaid
flowchart TB
  subgraph Bad["Bad current branch: constraint can be lost"]
    B1["Spec processor exists"] --> B2["Processor ineligible<br/>for this round"]
    B2 --> B3["Build allow-all artifact"]
    B3 --> B4["has_active_processor = true"]
    B4 --> B5["Sampler skips processor"]
    B5 --> B6["Constraint silently lost"]
  end

  subgraph Good["Target branch: skip only when actually covered"]
    G1["Spec processor exists"] --> G2{"Spec verify<br/>applied?"}
    G2 -->|yes| G3["Add processor id to<br/>applied_processor_ids"]
    G3 --> G4["Sampler skips<br/>that processor only"]
    G2 -->|no| G5{"Fallback policy"}
    G5 -->|normal path| G6["Run processor in<br/>sampler preprocess"]
    G5 -->|disable/reject| G7["Controlled fallback<br/>or request error"]
  end

  classDef good fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
  classDef warn fill:#fff8e1,stroke:#f9a825,color:#5d4300
  classDef bad fill:#ffebee,stroke:#c62828,color:#7f0000
  class G3,G4,G6 good
  class G7 warn
  class B3,B4,B5,B6 bad
```

If a processor implements spec verify but is unavailable or ineligible for this round, the system must either:

- run its normal processor path,
- disable/fallback MTP for the request,
- or reject the request.

It must not generate an allow-all artifact and then skip the processor.

### Invariant 3: Commit update is explicit

Any processor whose state depends on committed tokens must be updated after token commit.

Do not use `isStateful()` for this. Introduce capability flags:

```cpp
struct ProcessorCapabilities {
    bool needs_commit_update = false;
    bool supports_spec_verify = false;
    bool supports_async_commit = false;
    bool validates_commit_length = false;
};
```

Examples:

| Processor | needs_commit_update | supports_spec_verify | validates_commit_length |
|---|---:|---:|---:|
| Grammar | yes | yes | yes |
| ThinkMode | yes | yes | yes or explicit no, but not default no-op |
| Tree | depends on behavior | no unless implemented | depends |
| MultiSeq | depends on behavior | no unless implemented | depends |

### Invariant 4: Processor state has a version

Each stream owns a processor state epoch:

```cpp
struct ProcessorStateEpoch {
    uint64_t version = 0;
    std::shared_ptr<torch::Event> commit_done_event;
};
```

Round N:

1. build spec artifact from processor state version `V`;
2. sample accepted tokens;
3. async worker commits stream tokens and all `needs_commit_update` processors;
4. processor state becomes version `V + 1`;
5. next round only waits when it needs to read processor state.

This keeps async overlap while preventing stale Grammar/ThinkMode snapshots.

## API Design

### Base processor interface

Add explicit capability methods:

```cpp
class BaseLogitsProcessor {
public:
    virtual ProcessorCapabilities capabilities() const;
    virtual void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) = 0;
    virtual void commit(const CommitContext& ctx) = 0;
    virtual int64_t acceptedTokenLen() const;
};
```

`commit()` replaces ambiguous update paths:

```cpp
struct CommitContext {
    enum class Phase {
        NormalDecode,
        MtpSpecCommit,
    };

    Phase phase;
    torch::Tensor committed_tokens;
    int32_t committed_token_count;
    torch::Tensor src_batch_indices;
    int64_t stream_output_len_after_commit;
};
```

### Spec verify interface

Replace the current `tryAcceptAndFillBitmask()` return contract with an explicit status:

```cpp
enum class SpecVerifyStatus {
    Applied,
    Noop,
    Unavailable,
    Error,
};

struct SpecVerifyResult {
    SpecVerifyStatus status;
    int cap;
};
```

Rules:

- `Applied`: mask/cap is valid and processor id is added to `applied_processor_ids`;
- `Noop`: processor has no constraint for this round and can be safely skipped only if it says so explicitly;
- `Unavailable`: fallback/reject path must be chosen;
- `Error`: request enters error state.

## ProcessorPlan

Create `ProcessorPlan` when a stream is created.

```cpp
struct ProcessorPlan {
    std::vector<BaseLogitsProcessorPtr> processors;
    std::vector<BaseLogitsProcessorPtr> commit_processors;
    std::vector<BaseLogitsProcessorPtr> spec_verify_processors;
    bool can_use_mtp = true;
    bool can_use_mtp_async = true;
};
```

Validation:

Compatibility is decided once when building the stream. The plan only asks three questions per processor:

1. Does this processor change state after committed tokens?
2. Is MTP requested for this stream?
3. If yes, can the processor be represented safely in spec verify and async commit?

```mermaid
flowchart TB
  A["Processor"] --> B{"needs_commit_update?"}
  B -->|no| C["No commit state<br/>normal sampler path is enough"]
  B -->|yes| D{"MTP requested?"}
  D -->|no| E["Normal decode commit"]
  D -->|yes| F{"supports_spec_verify<br/>and supports_async_commit?"}
  F -->|yes| G["Add to spec_verify_processors<br/>and commit_processors"]
  F -->|no| H{"Policy"}
  H -->|reject| I["Return request error"]
  H -->|disable_mtp| J["Use non-MTP decode"]
  H -->|disable_async| K["Use synchronous safe path"]

  classDef safe fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
  classDef warn fill:#fff8e1,stroke:#f9a825,color:#5d4300
  class C,E,G safe
  class I,J,K warn
```

Default policy should be correctness-first:

- production default: disable MTP async or reject unsupported constrained decoding;
- debug/perf experiments can opt into fallback via environment flag;
- every fallback emits metric and log.

## Constraint Normalization

Normalize all request grammar inputs into one canonical object before reaching C++ factory logic.

```cpp
struct ConstraintConfig {
    enum class Type {
        None,
        JsonSchema,
        Regex,
        Ebnf,
        StructuralTag,
    };

    Type type = Type::None;
    std::string payload;
};
```

Rules:

- `response_format: {"type":"text"}` produces `Type::None`;
- `json_format=true` produces JSON object schema only if no stronger grammar field exists;
- legacy fields are accepted only as input to normalization;
- `LogitsProcessorFactory` consumes `ConstraintConfig`, not raw field priority.

Current effective priority should be preserved during migration:

```text
json_schema > regex > ebnf > structural_tag > response_format
```

## MTP Async Decode Flow

This is the main runtime sequence. The important ownership rule is that `SpecVerifyArtifact` is produced by the spec worker, consumed by the sampler/cap path, and only reusable after the sampler stream records `artifact_consumed_event`.

```mermaid
sequenceDiagram
  autonumber
  participant Main as Main decode thread
  participant Spec as Spec logits worker
  participant Target as Target model
  participant Sampler as Sampler / cap path
  participant Commit as Bookkeeping worker
  participant Stream as GenerateStream

  Main->>Stream: read processor state version V
  Main->>Main: draftModelDecode produces draft tokens
  Main->>Spec: launch build artifact(draft tokens, version V)
  Main->>Target: run target verify forward
  Target-->>Sampler: target_logits_ready_event, or same-stream order
  Spec-->>Sampler: artifact_ready_event recorded
  Sampler->>Sampler: wait target_logits_ready_event if needed
  Sampler->>Sampler: wait artifact_ready_event
  Sampler->>Sampler: apply vocab_mask_gpu
  Sampler->>Sampler: run target sampling
  Sampler->>Sampler: rejection sampling + apply cap_gpu
  Sampler-->>Main: record artifact_consumed_event
  Sampler-->>Commit: record accepted_tokens_ready_event
  Main->>Stream: pass artifact_consumed_event to buffer pool
  Main->>Commit: launch async commit
  Commit->>Commit: wait accepted_tokens_ready_event
  Commit->>Stream: commit accepted tokens
  Commit->>Stream: commit all needs_commit_update processors
  Commit-->>Stream: publish processor state version V+1
  Main->>Stream: next round wants processor state
  alt version V+1 already published
    Stream-->>Main: read without blocking
  else commit still running
    Main->>Stream: wait commit_done_event, then read V+1
  end
```

Sampler preprocessing:

```mermaid
flowchart LR
  A[Processor in stream] --> B{ID is in artifact.applied_processor_ids?}
  B -->|yes| C[Skip: already covered by mask]
  B -->|no| D{Can run normal process in this phase?}
  D -->|yes| E[Run process or processSpeculative]
  D -->|no| F[Reject or fallback]

  style C fill:#d5f5d5,stroke:#2f7d32
  style F fill:#ffd6d6,stroke:#b71c1c
```

## Grammar and XGrammar Requirements

Grammar processor:

- `process()`, `commit()`, and spec verify all use the same `state_mutex_`, unless spec verify uses an immutable matcher snapshot;
- spec verify accept/rollback must be a transaction;
- each bitmask row must clear `[grammar_vocab, model_vocab)`;
- `terminated`, `finished`, and `passthrough` modes must be represented explicitly in spec verify results.

XGrammar backend:

- protect `compiler_.Compile*()` and `compiler_.ClearCache()` with compiler mutex;
- consider per-key in-flight compile futures to avoid stampede;
- distinguish invalid-cache miss from invalid-cache hit with empty error string.

## Migration Plan

### Phase 0: Guardrails

- Add logs/metrics for unsupported processor capability combinations.
- Add runtime assertion: if spec artifact exists, every skipped processor id must be in `applied_processor_ids`.
- Add assertion that artifact tensors are not reused before `artifact_consumed_event`.

### Phase 1: Processor capability layer

- Add `ProcessorCapabilities`.
- Implement capabilities for Grammar and ThinkMode.
- Add `needs_commit_update` path in `GenerateStream::update()` and `GenerateStream::specUpdate()`.
- Keep old APIs as compatibility wrappers.

### Phase 2: SpecVerifyArtifact ownership

- Replace raw `LaunchResult` tensors with `SpecVerifyArtifact`.
- Use owned tensors first for simplicity.
- Optionally optimize to ring/pool after correctness tests pass.

### Phase 3: Applied processor tracking

- Add stable processor ids or indices per stream.
- `SpecLogitsVerifyRunner` returns `applied_processor_ids`.
- `LogitsProcessorStates` skips only applied processors.

### Phase 4: Commit state version

- Add processor state epoch per stream.
- Async commit worker records commit done event.
- Next round waits only on processor-state reads.

### Phase 5: Constraint normalization

- Add canonical `ConstraintConfig`.
- Update OpenAI, DashSC/RPC, and C++ factory path.
- Preserve legacy field parsing during migration.

### Phase 6: XGrammar and tests

- Add compiler mutex.
- Add full regression matrix.
- Split CPU-only grammar test deps.

## Regression Matrix

| Area | Required Case |
|---|---|
| Artifact lifetime | two consecutive MTP decode rounds reuse runner buffers; vary batch/propose/vocab |
| Applied semantics | all Spec processors ineligible; assert no processor skip occurs |
| ThinkMode commit | `accepted_len > 1` crosses think budget/end token; snapshot advances |
| Grammar vocab | grammar vocab smaller than model vocab; tail token logits are masked |
| Grammar concurrency | `tryAcceptAndFillBitmask()` interleaves with `commit()` under TSAN |
| Stateful non-Spec | MTP request rejects/fallbacks instead of silent skip |
| XGrammar backend | concurrent compile same/different grammar plus clear under TSAN |
| Config none | `response_format: text` clears all grammar sources including stale `response_format` |
| FakeSampler | test sampler calls preprocess or real sampler is used |
| BUILD | CPU grammar test has no CUDA deps |

## Rollout and Safety

Recommended rollout flags:

```text
RTP_LLM_MTP_ASYNC_LOGITS_PROCESSOR=0/1
RTP_LLM_MTP_UNSUPPORTED_PROCESSOR_POLICY=reject|disable_mtp|disable_async
RTP_LLM_MTP_SPEC_ARTIFACT_POOL=owned|ring
```

Default policy before full coverage:

```text
RTP_LLM_MTP_ASYNC_LOGITS_PROCESSOR=0
RTP_LLM_MTP_UNSUPPORTED_PROCESSOR_POLICY=reject
RTP_LLM_MTP_SPEC_ARTIFACT_POOL=owned
```

Promotion criteria:

- all P0/P1 regression cases pass;
- no silent skip metrics in canary;
- no processor state length mismatch;
- no grammar invalid-token commit errors under MTP async canary;
- no TSAN failures for grammar matcher/backend tests.

## Acceptance Checklist

- [ ] `SpecVerifyArtifact` has explicit ownership and consumed event.
- [ ] `LogitsProcessorStates` skips only applied processor ids.
- [ ] Grammar and ThinkMode declare commit/update capabilities.
- [ ] MTP commit updates all `needs_commit_update` processors.
- [ ] Processor state epoch prevents stale snapshot reads.
- [ ] Stateful non-Spec processor policy is explicit.
- [ ] Grammar spec bitmask masks `[grammar_vocab, model_vocab)`.
- [ ] XGrammar compiler operations are mutex-protected.
- [ ] `response_format: text` normalizes to no constraint.
- [ ] FakeSampler or real sampler tests cover preprocess.
- [ ] CPU grammar test has CPU-only deps.
