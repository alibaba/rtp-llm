# 外部后端延迟注册机制

RTP-LLM 支持将设备相关后端放在开源目录之外，例如内源后端或第三方扩展。
这些后端可能需要扩展 Linear、Fused MoE、Attention，或者向命令行参数中增加
后端特有的 MoE 策略，但不应将设备相关导入直接写入公共 Factory。

本文说明外部后端的延迟注册约定，包括入口加载、注册时序、槽位生命周期、公开
接口、扩展方式、失败语义、兼容性边界和测试范围。本机制是独立的运行时扩展能力，
不包含任何具体量化格式、权重加载流程或设备 kernel。

## 为什么需要延迟注册

外部后端入口需要先登记扩展意图，但入口导入阶段不能直接调用
`LinearFactory.register()` 或 `StrategyRegistry.register()`，因为对应 Factory
可能尚未创建。反过来，如果外部入口为了注册实现而提前导入 Fused MoE 或 Attention
Factory，又会连带加载通信库和设备库，影响本来不使用这些 Factory 的进程。

因此，外部后端入口只在指定槽位中登记注册函数：

```python
from rtp_llm.utils.backend_registry import register_backend_hook


register_backend_hook("linear", register_linear_impl)
```

公共 Factory 在原生 registry 或实现列表准备完成后，再执行该槽位中的注册函数：

```python
run_backend_registrations("linear", factory=LinearFactory)
```

所有 `run_backend_registrations()` 调用都会先通过稳定的 `models_py` 入口加载
可选后端，再读取和执行槽位中的 hook。因此即使业务代码先导入某个公共 Factory，
外部入口也能在该 Factory 消费槽位之前完成登记。`setup_args()` 还会在创建 parser
和参数组之前显式加载同一个入口，保证参数扩展遵循相同顺序。

外源环境中没有 `internal_source` 时，入口加载返回 `False`，槽位没有 hook 时
正常 no-op，不改变现有 CUDA 或 ROCm 路径。

## 公开接口

注册接口位于 `rtp_llm.utils.backend_registry`：

```python
ensure_backend_entrypoint_loaded()
register_backend_hook(slot, hook)
run_backend_registrations(slot, repeatable=False, **context)
```

`ensure_backend_entrypoint_loaded()` 只依赖稳定的可选入口名，不依赖任何具体内源
模块。入口导入本身带缓存，同一进程中的重复调用不会重复执行入口模块。

`register_backend_hook()` 按登记顺序保存 hook。某个槽位开始执行后，hook 集合即
冻结；再向该槽位登记 hook 会抛出 `RuntimeError`，避免不同 Factory 或 parser
实例看到不一致的扩展集合。

`run_backend_registrations()` 先确保入口已加载，再使用关键字参数将槽位上下文传给
hook。默认生命周期是一次性：同一槽位最多执行一次，后续调用是 no-op。设置
`repeatable=True` 时，首次执行仍会冻结 hook 集合，但以后每次调用都会把同一组
hook 重放到新的 owner 上。一个槽位开始后不能在一次性和可重复生命周期之间切换。

hook 抛出的异常不会被吞掉。后端声明了某项注册但注册失败时，启动过程应立即失败，
不能静默回退到其他实现，否则可能把配置错误延迟到 kernel 执行阶段，甚至产生错误
计算结果。

`reset_backend_registrations()` 用于清理进程级注册状态，只供单元测试使用。

## 扩展槽位

| 槽位 | 生命周期 | 执行位置 | 传给 hook 的上下文 | 用途 |
| --- | --- | --- | --- | --- |
| `linear` | 一次性 | 原生 `LinearFactory` 实现导入完成后 | `factory` | 注册外部 `LinearBase` 实现 |
| `fused_moe` | 一次性 | 设备侧 `StrategyRegistry` 填充并安装完成后 | `registry` | 注册外部 Fused MoE 策略 |
| `attention` | 一次性 | 原生 Attention 实现列表填充完成后 | `prefill_mha_imps`、`decode_mha_imps`、`prefill_mla_imps`、`decode_mla_imps` | 按优先级插入外部 Attention 实现 |
| `moe_strategy_choices` | 每个 parser 重放 | 公共 parser 创建 `--moe_strategy` 参数后 | `parser` | 增加外部后端提供的 MoE 策略名称 |

Attention 实现列表的顺序就是选择优先级，越靠前的实现越先参与匹配。hook 可以直接
操作四个实现列表，根据后端能力将实现插入到合适位置。公共注册机制只提供列表和
生命周期，不规定具体后端的优先级策略。

`moe_strategy_choices` 必须对每个新 parser 重放，因为 `setup_args()` 可以在
同一进程中调用多次，每次都会创建新的 parser 和新的 choices 列表。该槽位继续使用
`parser=` 上下文，现有外部 hook 不需要修改函数签名。

## 后端接入示例

后端入口只负责登记 hook。较重的实现模块可以放在回调内部导入，等公共 Factory
完成初始化并执行相应槽位时再加载：

```python
from rtp_llm.utils.backend_registry import register_backend_hook


def _register_linear(factory):
    from my_backend.linear import MyLinear

    factory.register(MyLinear)


def _register_moe(registry):
    from my_backend.moe import MyMoeStrategy

    registry.register(MyMoeStrategy())


def _extend_moe_choices(parser):
    for action in parser._actions:
        if "--moe_strategy" in action.option_strings:
            choices = list(action.choices or ())
            if "my_backend_strategy" not in choices:
                choices.append("my_backend_strategy")
            action.choices = choices
            return
    raise RuntimeError("--moe_strategy 尚未初始化")


register_backend_hook("linear", _register_linear)
register_backend_hook("fused_moe", _register_moe)
register_backend_hook("moe_strategy_choices", _extend_moe_choices)
```

入口层注册应具备幂等性，因为同一个入口可能由参数解析或不同 Factory 的导入路径
触达。公共入口加载带缓存，后端仍应避免自行调用安装函数时重复登记同一个 callable。

外部入口在登记阶段不应导入公共 Factory。具体实现模块应放在 hook 内部延迟导入，
这样 Factory 加载入口时不会形成循环依赖。

## 失败语义

注册机制采用 fail-fast 行为：

- 槽位开始执行后再登记 hook，抛出 `RuntimeError`；
- 槽位开始后改变一次性或可重复生命周期，抛出 `RuntimeError`；
- hook 内部异常直接向上传递；
- 没有 hook 的槽位正常执行并成为 no-op；
- 一次性槽位重复执行不会重复注册实现；
- 可重复槽位对每个 owner 重放首次冻结的 hook 集合；
- 不同槽位之间相互隔离，并分别维护执行状态。

注册状态由进程级锁保护，避免并发导入或并发初始化时重复消费一次性槽位。这些约束
用于避免后端注册失败后继续选择不兼容的公开实现。

## 修改文件与职责

本机制的修改按照公共模块的所有权组织：

| 文件 | 职责 |
| --- | --- |
| `rtp_llm/utils/backend_registry.py` | 加载可选入口、保存和冻结 hook、管理一次性或可重复生命周期、拒绝过晚注册，并提供测试清理接口 |
| `rtp_llm/models_py/modules/factory/linear/__init__.py` | 在原生 Linear 实现注册后执行 `linear` 槽位 |
| `rtp_llm/models_py/modules/factory/fused_moe/__init__.py` | 在设备侧 registry 安装后执行 `fused_moe` 槽位 |
| `rtp_llm/models_py/modules/factory/attention/__init__.py` | 使用四个有序实现列表执行 `attention` 槽位 |
| `rtp_llm/server/server_args/moe_group_args.py` | 在公共 CLI 参数创建后，对每个 parser 执行 `moe_strategy_choices` 槽位 |
| `rtp_llm/server/server_args/server_args.py` | 在创建 parser 和参数组之前确保可选后端入口已加载 |
| `rtp_llm/utils/test/backend_registry_test.py` 及其 `BUILD` target | 验证入口顺序、registry 状态、生命周期和错误约定 |
| `rtp_llm/server/server_args/test/server_args_test.py` | 验证连续创建的 parser 都能解析外部 MoE 策略 |
| `docs/backend/backend_registration.md` 和 `docs/index.rst` | 发布并索引外部后端接入约定 |

## 兼容性与职责边界

所有 Factory 槽位仍位于现有公共 registry 初始化完成之后。没有外部后端登记 hook
时，执行槽位只是 no-op，现有 CUDA 和 ROCm 的实现加载、优先级和选择逻辑保持
不变。

本次生命周期修复保留已有的槽位名和关键字上下文，包括
`factory=`、`registry=`、四个 Attention 列表以及 `parser=`。外部后端不需要
修改现有 hook 函数签名。公共代码负责入口顺序、槽位名称、执行时机和上下文；外部
后端负责具体实现、能力检查以及自身实现之间的优先级。

`backend_registry.py` 必须保持设备和厂商无关，不应加入 PPU、CUDA、ROCm 或其他
后端的具体实现名称和导入。

本注册机制不包含以下内容：

- 具体设备的 Linear、Fused MoE 或 Attention 实现；
- 权重格式、权重加载和量化配置解析；
- GEMM、activation quantization 或其他设备 kernel；
- 设备能力判断、实现优先级和 warmup 策略；
- 具体外部后端的正确性、性能和端到端验证。

本机制只定义公共扩展点及其生命周期。具体外部后端按需登记 hook，并自行负责实现
选择、能力判断和运行时验证。

## 测试覆盖

公共 registry 的定向测试覆盖：

- 消费槽位前加载可选入口，入口能够及时登记 hook；
- hook 延迟执行和关键字上下文传递；
- 多个 hook 的登记顺序；
- 不同槽位之间的状态隔离；
- 一次性槽位最多执行一次；
- 可重复槽位对不同 owner 重放同一组 hook；
- 过晚注册和生命周期切换时抛出异常；
- hook 异常向上传递；
- 空槽位 no-op；
- 连续两次 `setup_args()` 都能解析外部 MoE 策略。

具体外部后端还需要独立验证：hook 是否注册了预期实现、Factory 是否能在对应设备
能力条件下选择该实现，以及 kernel 等价性、分布式行为、性能和端到端推理结果。
