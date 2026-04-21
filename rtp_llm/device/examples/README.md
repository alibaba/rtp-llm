# RTP-LLM 新设备接入指南

## 概述

RTP-LLM 使用两层架构来管理设备特定逻辑：

```
Device (优先级路由)      — 设备子类返回实现类列表，声明"我优先用什么"
    ↓
Selector (选择器)        — 框架代码，遍历列表 → 调 support() → 选中
```

新设备只需要改动 **2 个文件**（设备类 + `__init__.py` 注册），不需要修改任何 factory 或 selector 代码。

---

## 接入步骤

### Step 1: 实现你的算子和 attention

在对应目录下创建你的设备实现：

```
rtp_llm/models_py/modules/
├── base/your_device/           # norm, activation, topk 等基础算子
├── factory/attention/your_device_impl/  # attention 实现
├── factory/linear/impl/your_device/     # linear 实现
└── factory/fused_moe/impl/your_device/  # MoE 策略（如果支持）
```

每个 attention 实现必须继承 `FMHAImplBase` 并实现 `support()` 方法，如有禁用开关可 override `is_available()`：

```python
class YourPrefillImpl(FMHAImplBase):
    @classmethod
    def is_available(cls, fmha_config) -> bool:
        # 可选：关联你的 fmha_config 开关。不 override 则默认 True。
        return fmha_config is None or fmha_config.your_flag

    @staticmethod
    def support(attn_configs, attn_inputs) -> bool:
        # 声明自己能处理什么配置
        return attn_inputs.is_prefill

    def forward(self, qkv, kv_cache, layer_idx=0):
        # 你的 attention 实现
        ...
```

### Step 2: 创建设备类

在 `rtp_llm/device/` 下创建你的设备文件（如 `your_device.py`）：

```python
class YourDeviceImpl(DeviceBase):
    # Attention: 在方法体内 import，返回实现类列表
    def get_prefill_mha_priorities(self):
        from rtp_llm.models_py.modules.factory.attention.your_device_impl import YourPrefillImpl
        return [YourPrefillImpl]

    def get_decode_mha_priorities(self):
        from rtp_llm.models_py.modules.factory.attention.your_device_impl import YourDecodeImpl
        return [YourDecodeImpl]

    # Base Ops: import 实际类，返回 BaseOps NamedTuple
    def get_base_ops(self):
        from rtp_llm.device.base_ops import BaseOps
        from rtp_llm.models_py.modules.base.your_device.norm import RMSNorm, ...
        from rtp_llm.models_py.modules.base.your_device.activation import FusedSiluAndMul
        return BaseOps(
            RMSNorm=RMSNorm,
            FusedSiluAndMul=FusedSiluAndMul,
            # ... 完整列表参考 BaseOps 定义
        )

    # Linear: import 模块触发策略注册
    def register_linear_impl(self):
        import rtp_llm.models_py.modules.factory.linear.impl.your_device  # noqa: F401

    # MoE: 返回策略类列表
    def get_moe_strategy_candidates(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.your_device.strategy import YourStrategy
        return [YourStrategy]
```

### Step 3: 注册设备类型（1 行）

编辑 `rtp_llm/device/__init__.py`：

```python
def get_device_cls(type: DeviceType):
    ...
    elif type == DeviceType.YourDevice:
        return YourDeviceImpl
```

### 完成!

你**不需要修改**以下文件：
- `factory/attention/__init__.py`
- `factory/attention/attn_factory.py`（选择器逻辑，禁用检查已下沉到各 impl 的 `is_available()`）
- `factory/linear/__init__.py`
- `factory/fused_moe/__init__.py`
- `modules/base/__init__.py`

---

## 工作原理

以 attention 选择为例：

1. 框架调用 `device.get_prefill_mha_priorities()` 拿到 `[YourPrefillImpl, ...]`
2. 遍历列表，调用 `YourPrefillImpl.support(config)` 检查兼容性
3. 通过检查则实例化并使用

设备方法内使用 lazy import（在方法体内 import），保证只有实际使用时才加载设备相关依赖。

---

## 完整示例

参考 `example_device_impl.py` 查看一个可运行的 dummy 设备实现。
