# compressed-tensors INT8 W8A8 识别（主线）

主线 `rtp_llm/config/quant_config.py` 现在可以直接识别 compressed-tensors
的 INT8 W8A8 checkpoint（per-channel 对称权重 + per-token 动态激活），
GLM-4.7 INT8 是这种格式的代表。本文说明识别逻辑、字段契约，以及与内源
PPU 实现的边界。

## 1. 字段契约

识别的唯一依据是 checkpoint `config.json` 里的 `quantization_config`：

```jsonc
{
  "quant_method": "compressed-tensors",
  "ignore": ["lm_head", "..."],          // 不量化的模块，可缺省
  "config_groups": {
    "group_0": {
      "weights":           {"num_bits": 8, "type": "int", "strategy": "channel", "symmetric": true},
      "input_activations": {"num_bits": 8, "type": "int", "strategy": "token",   "dynamic": true}
    }
  }
}
```

契约在 `W8A8Int8PerChannelCompressedQuantConfig` 上以两个类常量表达，
逐字段严格相等比较：

| 段 | 字段 | 期望值 |
|----|------|--------|
| `weights` | `num_bits` / `type` / `strategy` / `symmetric` | `8` / `"int"` / `"channel"` / `True` |
| `input_activations` | `num_bits` / `type` / `strategy` / `dynamic` | `8` / `"int"` / `"token"` / `True` |

`type` 的期望值取自主线已有的 `QuantizationType.INT`，不新造字符串常量。
`strategy` 沿用周围代码的裸字符串写法（`"channel"` / `"tensor"` / `"group"`），
没有为它引入新枚举。

激活是动态量化，scale 在运行时按 token 现算，checkpoint 里**没有**
`input_scale`；权重侧只有 `.weight_scale`（`weight_scale_suffix`）。

## 2. 识别流程

沿用主线既有的两段式范式，没有另起炉灶：

1. **注册**：`QuantizationConfig.__init_subclass__` 把每个子类按类名登记进
   `_registry`；`from_config()` 用 `get_method()` 做大小写无关匹配。新类的
   method 是 `W8A8_INT8_PER_CHANNEL_COMPRESSED`，与 `preset_quant_config`
   的新增 key 同名，因此 `--quantization W8A8_INT8_PER_CHANNEL_COMPRESSED`
   和 JSON 形式的 `{"method": ...}` 两条入口都能拿到同一个类。
2. **识别**：`load_from_ckpt()` 的 `compressed-tensors` 分支新增一个 `elif`，
   调用 `matches_weights()` 做分支判别，命中后交给
   `from_checkpoint_quant_config()` 做完整校验并构造。

判别与校验刻意分成两步，对应两类不同的语义：

- `matches_weights()` 只看 **dtype / 位宽 / 粒度** 这三个"这是哪套方案"的字段。
  不命中说明 checkpoint 属于别的方案（FP8 per-channel、INT4 group……），
  本分支不认领、也不代它报错，继续走原有分支。
- `from_checkpoint_quant_config()` 在已经认领之后做**全字段校验**，任何一个
  字段不符就抛 `ValueError`，并在消息里带上 `{字段: (实际值, 期望值)}`。

**fail-closed 的理由**：主线的兜底路径会用 `method="compressed-tensors"` 落到
基类 `CompressedTensorsQuantConfig`，而这个基类没有实现
`get_supported_compute_dtypes` / `get_supported_kv_cache_dtypes` 两个抽象方法，
实例化时直接 `TypeError`（这是本次改动之前就存在的行为，本次没有去动它）。
也就是说，一个"看起来像 INT8 W8A8 但某个字段不对"的 checkpoint，
过去不会被静默误用，而是会在更深的地方以难以归因的方式炸掉。现在它在解析阶段
就拿到一条指明哪个字段不对的错误。

顺带的结论是：这套格式在本次改动之前，主线**完全无法加载**（既没有对应分支，
兜底又是抽象基类），所以新增分支不会改变任何原本可用的 checkpoint 的行为。
`test_fp8_per_channel_still_wins` / `test_int4_group_still_wins` 用来守住这一点。

## 3. `ignore` 与 `exclude_modules`

compressed-tensors 用顶层 `ignore` 列出不量化的模块（`lm_head` 等）。
构造时同时写入两处：

- `ignore_patterns` —— compressed-tensors 自己的叫法，与 Kimi-K2.5 的
  `CompressedW4A8Int4PerChannelQuantConfig` 保持一致；
- `exclude_modules` —— `QuantizationConfig` 基类的通用字段，权重加载侧
  （如 `per_channel_fp8_quant_weight.py`）读的是它。

`load_from_ckpt()` 结尾那段 `quant_config["exclude"]` 是另一种 checkpoint 的
拼写，与 `ignore` 不是同一个 key；本分支提前 return，不走那段。

## 4. 与内源 PPU 实现的边界

内源 `internal_source/rtp_llm/models_py/glm47_int8_w8a8/core/quant_config.py`
里的 `parse_checkpoint_quantization()` 是本次字段校验逻辑的权威依据，
两边的**字段契约完全一致**。但内源那个类还带着一批 PPU 专有的东西，
**都不在主线这份实现里**：

| 内源专有 | 主线不做的原因 |
|----------|----------------|
| `get_supported_kv_cache_dtypes()` 只返回 `float8_e4m3fn` | 这是 PPU `Glm47KVCacheWriteOp` 硬依赖 native FP8 paged KV 的后果，是那条执行路径的约束，不是量化格式的约束。主线给的是通用的 `fp16 / bf16 / fp8_e4m3fn` |
| `get_supported_compute_dtypes()` 只返回 `bfloat16` | 同上，主线给通用的 `fp16 / bf16` |
| `get_algo()` 返回 `"smooth_quant"`（复用 C++ enum） | C++ enum 映射属于内源执行路径。主线返回描述性的 `"w8a8_int8_per_channel_compressed"`，只表达"这是哪套量化方案"，不对 kernel 绑定做承诺 |
| `install_runtime_quant_hooks()` / DeepEP LL 的 buffer 计算 | 纯 PPU 运行时策略 |
| `Glm47ModelConfig` / `RTP_LLM_GLM47_FP8_KV` / 模型选路 | PPU 部署形态 |

一句话：**主线只回答"这套量化元数据是什么、合不合法"，不回答"用哪个 kernel
跑、KV 怎么存"**。内源可以继续在自己的子类里覆写 dtype 与 algo，字段解析这一段
则可以直接复用主线的 `from_checkpoint_quant_config()`，不必再维护一份。

## 5. 测试

`rtp_llm/test/compressed_int8_quant_config_test.py`，纯 CPU、合成 dict、
不需要真实权重：

```bash
python3 -m unittest rtp_llm.test.compressed_int8_quant_config_test -v
```

bazel：

```bash
bazel test //rtp_llm/test:compressed_int8_quant_config_test
```

覆盖：正确组合被识别（含 `text_config` 嵌套形式）、`ignore` 的两处落点、
注册表与 preset 两条入口、以及 strategy 不符 / 激活非 dynamic / 激活非 8bit /
激活是 float / 权重非对称 / 结构残缺被正确报错，外加 FP8 per-channel 与
INT4 group 两条既有分支不被抢走。
