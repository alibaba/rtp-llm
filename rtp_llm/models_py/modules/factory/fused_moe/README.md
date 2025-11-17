# FusedMoe Factory Refactoring

This directory contains the refactored `FusedMoeFactory`, using strategy pattern and builder pattern to simplify MOE creation logic.

## Directory Structure

```
fused_moe/
├── __init__.py                   # Export main interface
├── factory.py                    # Main factory class
├── config_resolver.py            # Configuration resolver
├── quant_config.py               # Quantization configuration
├── type.py                       # RouterType and ExecutorType enums
├── README.md                     # This document
├── strategies/                   # Strategy infrastructure
│   ├── __init__.py
│   ├── base.py                   # Strategy base class
│   ├── condition_checker.py      # Reusable condition checker utility
│   ├── priority_attributes.py    # Priority calculation system
│   └── strategy_registry.py      # Strategy registry
└── tests/                        # Unit tests
    ├── BUILD
    ├── test_config_resolver.py
    ├── test_strategy_registry.py
    └── test_cuda_strategies.py
```

**Note**: Actual strategy implementations are located in:
- **CUDA strategies**: `rtp_llm/models_py/modules/cuda/moe/strategy/`
  - `no_quant.py`: Without quantization
  - `fp8_per_block.py`: FP8 PerBlock quantization
  - `fp8_per_tensor.py`: FP8 PerTensor quantization
- **ROCm strategies**: `rtp_llm/models_py/modules/rocm/moe/strategy/`
  - `ep.py`: Expert Parallelism strategies

Strategy registration is done in:
- `rtp_llm/models_py/modules/cuda_registry.py` (CUDA)
- `rtp_llm/models_py/modules/rocm_registry.py` (ROCm)

**DeepEpInitializer** is located at: `rtp_llm/models_py/distributed/deepep_initializer.py`

## Usage

### Basic Usage

```python
from rtp_llm.models_py.modules.factory.fused_moe import FusedMoeFactory

# Create FusedMoeFactory instance (registry is set automatically by cuda_registry.py or rocm_registry.py)
factory = FusedMoeFactory()

# Create FusedMoe instance
moe = factory.create_fused_moe(config, weights)
```

**Note**: The `FusedMoeFactory` registry is automatically initialized when the corresponding device module is imported:
- CUDA: Importing `rtp_llm.models_py.modules.cuda_registry` sets up the CUDA registry
- ROCm: Importing `rtp_llm.models_py.modules.rocm_registry` sets up the ROCm registry

The interface is fully compatible with the old version, no need to modify existing code.

## Design Patterns

### 1. Strategy Pattern

Each hardware and configuration combination has a corresponding strategy class:

- **CUDA Strategies** (located in `rtp_llm/models_py/modules/cuda/moe/strategy/`):
  - `CudaNoQuantEpLowLatencyStrategy`: EP low latency without quantization
  - `CudaFp8PerBlockNoDPStrategy`: FP8 PerBlock without DP
  - `CudaFp8PerBlockEpLowLatencyStrategy`: FP8 PerBlock EP low latency
  - `CudaFp8PerBlockEpNormalStrategy`: FP8 PerBlock EP normal
  - `CudaFp8PerTensorSingleGpuStrategy`: FP8 PerTensor single GPU
  - `CudaFp8PerTensorEpLowLatencyStrategy`: FP8 PerTensor EP low latency
  - `CudaFp8PerTensorEpNormalStrategy`: FP8 PerTensor EP normal
  - `BatchedTritonStrategy`: Fallback strategy using Triton

- **ROCm Strategies** (located in `rtp_llm/models_py/modules/rocm/moe/strategy/`):
  - `RocmEpNormalStrategy`: EP normal mode
  - `RocmEpLowLatencyStrategy`: EP low latency mode
  - `BatchedTritonStrategy`: Fallback strategy using Triton

### 2. Registry Pattern

`StrategyRegistry` automatically manages and selects strategies, sorted by priority.

**Registration**: Strategies are registered in device-specific registry files:
- CUDA: `rtp_llm/models_py/modules/cuda_registry.py`
- ROCm: `rtp_llm/models_py/modules/rocm_registry.py`

Each registry creates a `StrategyRegistry` instance, registers all relevant strategies, and sets it on `FusedMoeFactory` using `FusedMoeFactory.set_registry()`.

## Core Components

### MoeConfigResolver

Configuration resolver, provides the following methods:

- `get_device_type()`: Get device type
- `has_quantization(config)`: Check if quantization is enabled
- `is_bf16(config)`: Check if data type is bf16
- `get_quant_method(config)`: Get quantization method
- `is_ep_enabled(config)`: Check if EP is enabled
- `use_low_latency(config)`: Check if low latency mode is used
- `is_single_gpu(config)`: Check if single GPU
- `is_tp_equal_ep(config)`: Check if TP equals EP

### FusedMoEQuantConfig

Quantization configuration dataclass (`quant_config.py`), provides:
- `quant_dtype`: Post-quantization activation type
- `per_act_token_quant`: Per-activation token quantization flag
- `per_out_ch_quant`: Per-output channel quantization flag
- `block_shape`: Block shape for block quantization
- Properties: `is_quantized`, `is_per_act_token`, `is_block_quantized`, `is_per_tensor`
- Methods: `scale_shape()`, `batched_scale_shape()`

### RouterType and ExecutorType

Type enums (`type.py`) for priority calculation:
- **RouterType**: `BATCHED_DATA` (0), `DEEPGEMM_CONTINUOUS` (1), `DEEPEP_NORMAL` (2), `DEEPEP_LOW_LATENCY` (4), `PURE_TP` (5)
- **ExecutorType**: `BATCHED_TRITON` (0), `DEEPGEMM_CONTINUOUS` (1), `DEEPGEMM_MASKED` (2), `FUSED_MOE` (2), `CUTLASS_FP8` (3), `CUTLASS_BATCHED_FP8` (4)

### DeepEpInitializer

Thread-safe DeepEP initialization manager (located at `rtp_llm/models_py/distributed/deepep_initializer.py`), ensures initialization only happens once.

### MoeStrategy

Base class for all strategies, defines the following interface:

- `can_handle(config)`: Determine if this configuration can be handled (automatically calls Router and Executor's `check_conditions`)
- `create_router(config)`: Create Router (handles DeepEP initialization internally if needed)
- `create_executor(config, weights)`: Create Executor
- `get_attributes()`: Return strategy attributes (Router and Executor types)
- `priority`: Strategy priority (automatically calculated from attributes)

### ConditionChecker

Reusable utility class for condition checking with automatic logging:

- Automatically extracts condition expressions from source code
- Records all condition check results
- Supports multi-line condition expressions
- Provides detailed debug logging with ✓/✗ symbols
- Can be used by any module, not just strategies

### Condition Checking Architecture

**Decentralized condition checking**: Each Router and Executor class defines its own conditions:

```python
class MyRouter(FusedMoeDataRouter):
    @classmethod
    def check_conditions(cls, checker: ConditionChecker, config: GptInitModelParameters) -> None:
        """Check if this router can handle the configuration"""
        resolver = MoeConfigResolver()
        checker.check(resolver.is_ep_enabled(config))
        checker.check(resolver.use_low_latency(config))
```

**Benefits**:
- **Single Responsibility**: Each component checks its own requirements
- **Reduced Duplication**: Conditions are defined once in the component
- **Better Maintainability**: Modify component conditions in one place
- **Clear Dependencies**: Easy to see what each component requires

## Adding New Strategies

### File Organization

1. Choose the appropriate directory:
   - **CUDA strategies**: `rtp_llm/models_py/modules/cuda/moe/strategy/`
     - No quantization → `no_quant.py`
     - FP8 PerBlock → `fp8_per_block.py`
     - FP8 PerTensor → `fp8_per_tensor.py`
   - **ROCm strategies**: `rtp_llm/models_py/modules/rocm/moe/strategy/`
     - Expert Parallel → `ep.py`

2. Create new strategy class, inherit from `MoeStrategy` (from `rtp_llm.models_py.modules.factory.fused_moe.strategies.base`)
3. Implement all required methods: `create_router()`, `create_executor()`, `get_attributes()`
4. Export in the strategy directory's `__init__.py`
5. Register in the device-specific registry file:
   - CUDA: `rtp_llm/models_py/modules/cuda_registry.py`
   - ROCm: `rtp_llm/models_py/modules/rocm_registry.py`

### Example: Adding a New Strategy

#### Step 1: Implement Router with conditions

```python
# In rtp_llm/models_py/modules/cuda/moe/routers/my_router.py
from typing import Any
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.common.moe.fused_moe import FusedMoeDataRouter
from rtp_llm.models_py.modules.factory.fused_moe.type import RouterType

class MyRouter(FusedMoeDataRouter):
    @classmethod
    def router_type(cls) -> RouterType:
        return RouterType.DEEPEP_LOW_LATENCY

    @classmethod
    def check_conditions(cls, checker: Any, config: GptInitModelParameters) -> None:
        """Check if MyRouter can handle the configuration"""
        from rtp_llm.models_py.modules.factory.fused_moe.config_resolver import MoeConfigResolver
        resolver = MoeConfigResolver()
        # Define router-specific conditions
        checker.check(resolver.is_ep_enabled(config))
        checker.check(resolver.use_low_latency(config))

    def __init__(self, config: GptInitModelParameters):
        # Implementation...
```

#### Step 2: Implement Executor with conditions

```python
# In rtp_llm/models_py/modules/cuda/moe/executors/my_executor.py
from typing import Any, Dict
import torch
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.common.moe.fused_moe import FusedMoeExpertExecutor
from rtp_llm.models_py.modules.factory.fused_moe.quant_config import FusedMoEQuantConfig
from rtp_llm.models_py.modules.factory.fused_moe.type import ExecutorType

class MyExecutor(FusedMoeExpertExecutor):
    @classmethod
    def executor_type(cls) -> ExecutorType:
        return ExecutorType.CUTLASS_BATCHED_FP8

    @classmethod
    def check_conditions(cls, checker: Any, config: GptInitModelParameters) -> None:
        """Check if MyExecutor can handle the configuration"""
        from rtp_llm.models_py.modules.factory.fused_moe.config_resolver import MoeConfigResolver
        resolver = MoeConfigResolver()
        # Define executor-specific conditions
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "MY_QUANT")

    def __init__(self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]):
        super().__init__(FusedMoEQuantConfig())
        # Implementation...
```

#### Step 3: Create Strategy (no condition checking needed!)

```python
# In rtp_llm/models_py/modules/cuda/moe/strategy/my_quant.py
from typing import Dict
import torch
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.factory.fused_moe.strategies.base import MoeStrategy
from rtp_llm.models_py.modules.factory.fused_moe.strategies.priority_attributes import StrategyAttributes

class CudaMyQuantStrategy(MoeStrategy):
    # No need to implement can_handle() or _check_conditions()!
    # Conditions are automatically checked via Router and Executor classes

    def create_router(self, config: GptInitModelParameters):
        from rtp_llm.models_py.modules.cuda.moe.routers.my_router import MyRouter
        return MyRouter(config)

    def create_executor(self, config: GptInitModelParameters,
                       weights: Dict[str, torch.Tensor]):
        from rtp_llm.models_py.modules.cuda.moe.executors.my_executor import MyExecutor
        return MyExecutor(config, weights)

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.cuda.moe.routers.my_router import MyRouter
        from rtp_llm.models_py.modules.cuda.moe.executors.my_executor import MyExecutor

        # Define Router and Executor classes
        # Priority is calculated automatically from router_type() and executor_type() methods
        # Conditions are checked automatically via router_class and executor_class
        return StrategyAttributes(
            router_class=MyRouter,
            executor_class=MyExecutor,
        )
```

#### Step 4: Register the Strategy

```python
# In rtp_llm/models_py/modules/cuda_registry.py
from rtp_llm.models_py.modules.cuda.moe.strategy.my_quant import CudaMyQuantStrategy
from rtp_llm.models_py.modules.factory.fused_moe import FusedMoeFactory, StrategyRegistry

registry = StrategyRegistry()
# ... register other strategies ...
registry.register(CudaMyQuantStrategy())
FusedMoeFactory.set_registry(registry)
```

**Key Points**:
- ✅ **No `can_handle()` override needed**: Base class handles it automatically
- ✅ **Conditions in components**: Router and Executor define their own requirements
- ✅ **Automatic checking**: `can_handle()` calls component `check_conditions()` automatically
- ✅ **Reduced duplication**: Conditions defined once per component, reused across strategies

### Priority System

Priority is **automatically calculated** based on Router and Executor types:
- **Formula**: `priority = router_type.value * 10 + executor_type.value`
- **Router types**: `BATCHED_DATA` (0) < `DEEPGEMM_CONTINUOUS` (1) < `DEEPEP_NORMAL` (2) < `DEEPEP_LOW_LATENCY` (4) < `PURE_TP` (5)
- **Executor types**: `BATCHED_TRITON` (0) < `DEEPGEMM_CONTINUOUS` (1) < `DEEPGEMM_MASKED`/`FUSED_MOE` (2) < `CUTLASS_FP8` (3) < `CUTLASS_BATCHED_FP8` (4)

**Examples**:
- `BATCHED_DATA` + `BATCHED_TRITON` = 0×10 + 0 = **0** (lowest)
- `DEEPEP_LOW_LATENCY` + `CUTLASS_BATCHED_FP8` = 4×10 + 4 = **44** (highest)
- `DEEPEP_NORMAL` + `CUTLASS_FP8` = 2×10 + 3 = **23** (mid-high)
- `PURE_TP` + `CUTLASS_BATCHED_FP8` = 5×10 + 4 = **54** (highest possible)

This means you **don't need to manually set numeric priorities**. Just declare what Router and Executor implementations your strategy uses (via `get_attributes()`), and the priority is calculated automatically based on their performance characteristics. Router and Executor classes must implement `router_type()` and `executor_type()` class methods that return the corresponding enum values.

## Testing

Run unit tests:

```bash
cd <project_root>
python -m pytest rtp_llm/models_py/modules/factory/fused_moe/tests/
```

Or using Bazel:

```bash
bazelisk test //rtp_llm/models_py/modules/factory/fused_moe/tests:all
```

## Refactoring Benefits

### Architecture Benefits
1. **Readability**: Each strategy class is only responsible for one scenario, code is clear
2. **Maintainability**: Modifying specific scenarios only requires modifying corresponding strategies
3. **Extensibility**: Adding new devices/quantization methods only requires adding new strategies and registration
4. **Testability**: Each component can be tested independently
5. **Thread safety**: DeepEP initialization uses singleton pattern

### Condition Checking Benefits
6. **Single Responsibility**: Each Router/Executor checks its own requirements
7. **Reduced Duplication**: Conditions defined once in components, not repeated in strategies
8. **Automatic Extraction**: Condition expressions extracted from source code for better logging
9. **Multi-line Support**: Handles condition expressions that span multiple lines
10. **Reusable**: `ConditionChecker` can be used by any module

### Example Logging Output

```
[ConditionChecker] Checking CudaFp8PerTensorEpLowLatencyStrategy.can_handle():
  ✗ condition_1: resolver.is_ep_enabled(config) = False
  ✓ condition_2: resolver.use_low_latency(config) = True
  ✗ condition_3: DeepEpInitializer.supported() = False
  ✓ condition_4: quant_method in ["FP8_PER_TENSOR_COMPRESSED", "FP8_DYNAMIC_PER_TENSOR"] = True
  → Final result: False
```
