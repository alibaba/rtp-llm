---
name: hot-hook-debugging
description: "RTP-LLM Python 热 hook 与 tensor dump 调试：Hot Hook 热更新、常驻 smoke/server 调试、function/line hook、A/B replacement、精度对齐、算子替换验证、response 语义异常定位、fused/JIT/Triton/TileLang/CUDA wrapper 输出对齐、不稳定问题排查、tensor diff。Use when: hot hook, dump tensor, tensor dump, precision alignment, operator replacement, A/B replacement, Python wrapper boundary, response mismatch, flaky debug."
---

# Hot Hook Debugging

本 skill 用于 RTP-LLM Python 层热 hook 和 tensor dump 调试。核心目标是：
**不重编译、不重载模型、在同一个 backend pid 内热更新 hook/config**，快速完成
边界二分、A/B replacement、精度对齐和不稳定问题定位。

## 何时使用

- Python module function、class method、staticmethod 的入口、出口、异常或替换调试。
- Python wrapper 边界对齐：JIT/Triton/TileLang/CUDA pybind wrapper、fused op、旧路径 vs 新路径。
- response 语义异常：输出乱码、长度异常、NoneType、非法 token、routing/topk 集合异常。
- 精度对齐：old path / new path 的输入输出、hash、stats、bitwise/allclose diff。
- 算子替换验证：用 reference implementation 或保守路径临时替换疑似 backend。
- smoke/server 常驻调试：一次加载 weights，多次改 hook/config 并重复请求。
- flaky / nondeterminism 的局部定位：已缩小到 Python 层函数、wrapper 或具体文件行附近。

不要用它直接 hook CUDA/C++ 内部实现；只能在 Python 调用边界或 wrapper 处观察。
真正的 C++ crash、core dump、NCCL hang 从 server 日志与 core 文件直接分析。

## 基础环境

唯一开关是 hot hook 配置 JSON 路径（`--hot_hook_config` 参数或 `RTP_HOT_HOOK_CONFIG` 环境变量——路径非空即启用，无独立总开关）。JSON 同时承载 hook 定义与装载子参数（hook_file/dump_dir/allow_local_mutation）：

```bash
mkdir -p $PWD/.tmp/rtp_hot_hook/<case>/dumps

cat > $PWD/.tmp/rtp_hot_hook/<case>/config.json <<'EOF'
{
  "hook_file": "<case>/hooks.py 的绝对路径",
  "dump_dir": "<case>/dumps 的绝对路径",
  "case": "<case>",
  "function_hooks": [ ... ]
}
EOF

# 只有确实要修改 Python 局部变量时才加（默认不要加）：
#   "allow_local_mutation": true

export RTP_HOT_HOOK_CONFIG=$PWD/.tmp/rtp_hot_hook/<case>/config.json
```

四个旧 env（RTP_HOT_HOOK / RTP_HOT_HOOK_FILE / RTP_HOT_HOOK_DUMP_DIR / RTP_HOT_HOOK_ALLOW_LOCAL_MUTATION）与旧总开关已在 2026-11 的 env 清理中移除，不再生效；hook 文件与 dump 目录一律写进 config JSON。

推荐目录结构：

```text
$PWD/.tmp/rtp_hot_hook/<case>/
├── hooks.py
├── config.json
├── dumps/
└── ...
```

## 默认流程

1. **Function boundary 先行**：先包住疑似 layer、block、attention、moe、wrapper 的 Python forward。
   记录 shape、dtype、stride、stats、hash，只在必要时 dump tensor。
2. **从外到内二分**：先确认上游输入是否一致，再确认当前函数输出是否首次出现 diff。
3. **A/B replacement**：对疑似 backend/wrapper 临时替换为 reference/old path/conservative path，
   同时记录原始输入、替换输出和最终 response。
4. **Line hook 最后使用**：`sys.settrace` 开销明显，只对单个文件/行短时间启用。
5. **覆盖目标阶段**：`max_tokens=1` 多数只覆盖 prefill sampling；定位 first decode step 用
   `max_tokens=2`，既触发首个 decode，又避免长输出拖慢实验。
6. **每轮落盘**：保存 response、hook config、dump 路径和关键 compare 数字。不要只写“有差异”。

## Function Hook

`config.json`：

```json
{
  "case": "boundary_probe",
  "function_hooks": [
    {
      "target": "package.module.ClassOrFunction.forward",
      "before": "before_target",
      "after": "after_target"
    }
  ]
}
```

`hooks.py`：

```python
def before_target(ctx):
    x = ctx.args[1]
    ctx.note("input_stats", ctx.stats(x))
    ctx.dump("input", x)

def after_target(ctx):
    ctx.note("output_stats", ctx.stats(ctx.result))
    ctx.dump("output", ctx.result)
```

中文解释：function hook 适合做粗粒度边界确认。先判断“进函数前是否干净、出函数后是否
污染”，再决定是否继续往函数内部拆。

## Replacement Hook

用于隔离“当前 backend 是否引入差异”。replacement 返回非 `None` 时会跳过原函数：

```json
{
  "case": "backend_ab",
  "function_hooks": [
    {
      "target": "package.module.select_backend",
      "replace": "replace_with_reference"
    }
  ]
}
```

```python
def replace_with_reference(ctx):
    score = ctx.args[0]
    ctx.dump("backend_input", score)
    out = reference_impl(score, *ctx.args[1:], **ctx.kwargs)
    ctx.dump("reference_out", out)
    return out
```

中文解释：replacement hook 是最快的 A/B 验证方式。若替换为旧路径后 response 恢复，
说明问题在被替换边界内部或其输出契约；若仍异常，继续向上游或下游移动边界。

## Line Hook

先用 `nl -ba <file>` 确认行号。普通本地运行可以写绝对路径；Bazel test 下
`frame.f_code.co_filename` 通常是 runfiles 路径，优先写稳定的 `file_suffix`：

```json
{
  "case": "line_probe",
  "line_hooks": [
    {
      "file_suffix": "package/module/file.py",
      "line": 1234,
      "hook": "dump_locals"
    }
  ]
}
```

```python
def dump_locals(ctx):
    for name in ("x", "score", "out"):
        if name in ctx.locals:
            ctx.dump(name, ctx.locals[name])
```

中文解释：line hook 用于 function boundary 已不够细的情况，例如某个临时变量在函数内部
被 in-place 修改或某个分支返回了 `None`。它不适合长时间、大范围打开。

## Tensor Dump 原则

forward/hot path 内禁止：

```text
.cpu()、.item()、print(tensor)、torch.cuda.synchronize()、cudaStreamSynchronize
```

这些操作会插入 CPU-GPU 同步或 DMA，改变 CUDA stream 时序，可能掩盖或制造不稳定问题。

三种保存模式：

- `clone`：保存快照，后续 in-place 不会覆盖；会分配 GPU tensor，可能改变 allocator 状态。
- `view`：零 GPU 开销，共享 storage；后续 in-place 会看到被覆盖后的数据。
- `untyped_storage`：持有底层 storage，可对抗 `dispose_tensor`；需要同时记录 shape、stride、dtype。

推荐策略：

- 小 tensor（如 < 100K elements）完整 dump `.pt`，方便逐元素 diff。
- 大 tensor 先 dump hash/stats，只有定位到边界后再窄化保存范围。
- 如果 dump 改变复现率，降级为 `ctx.note(ctx.stats(...))`、hash、shape 或更窄边界。
- 统一在安全点保存：函数出口、forward 结束后、或 hook runtime 的 dump 机制；不要在核心 kernel
  发射前后插入手写同步。

## Diff 规范

每个结论必须写清比较口径和数字：

- bitwise：diff_count、max_abs、示例 index。
- allclose：atol/rtol、failed_count、max_abs、max_rel。
- set equality：集合是否一致，缺失/多余元素数量。
- order equality：集合相同但顺序是否变化。
- invalid/duplicate/out-of-range：非法值、重复值、越界值数量。
- score threshold：错误元素分数是否低于应选阈值。

TopK、routing、gather、sparse attention 这类集合语义结果，不能只看 elementwise diff。
集合正确但顺序不同，也可能让下游 gather/reduction 出现 bitwise drift。

## 热更新规则

- 每次 hook 命中前检查 hook/config 文件 mtime。
- hook/config reload 失败不会杀 server，会保留上一版并记录日志。
- config 写 `{ "enabled": false }` 会卸载 function monkeypatch。
- config 不含 line hooks 时会卸载 line trace。
- hook callback 抛错应只记录 error 并继续，避免 debug hook 杀死常驻 server。

## 实现位置

运行时实现在 `rtp_llm/utils/hot_hook_runtime.py`；四个服务入口
（`start_server.py` / `start_backend_server.py` / `start_frontend_server.py` /
`start_dash_sc_server.py`）在各自参数解析后安装。单元测试
`rtp_llm/utils/test/hot_hook_runtime_test.py` 覆盖了完整行为（JSON fixture 驱动）。
