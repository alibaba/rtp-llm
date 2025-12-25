# Modules Development Guide

## Overview

The `modules` directory contains the core building blocks for model implementations in RTP-LLM. All operations within this directory are organized into three distinct categories to ensure modularity, maintainability, and architecture-specific optimization.

## Module Categories

### 1. Base Modules (`base/`)

Base modules are independent, self-contained modules without dependencies on other modules within the `modules` directory. These modules may have different implementations across various hardware architectures.

**Characteristics:**
- Self-contained with no dependencies on other module types (factory or hybrid)
- Architecture-specific implementations (e.g., CUDA, ROCm)
- Provide fundamental operations such as normalization, activation, and embedding layers
- Architecture selection is handled automatically based on device type detection

**Directory Structure:**
```
base/
├── common/           # Architecture-independent implementations
│   ├── embedding.py
│   ├── kvcache_store.py
│   └── norm.py
├── cuda/            # CUDA-specific implementations
│   ├── activation.py
│   ├── norm.py
│   └── select_topk.py
└── rocm/            # ROCm-specific implementations
    ├── activation.py
    ├── norm.py
    ├── select_topk.py
    └── not_implemented_ops.py
```

**Examples:**
- `Embedding`: Embedding layer implementations
- `RMSNorm`, `LayerNorm`: Normalization operations
- `SelectTopk`, `GroupTopK`: Selection operations
- `WriteCacheStoreOp`: KV cache storage operations

### 2. Factory Modules (`factory/`)

Factory modules encapsulate the logic for selecting and instantiating appropriate implementations based on configuration and architecture. They hide the complexity of choosing the right implementation from the end user.

**Characteristics:**
- Implement the factory pattern for module instantiation
- Select implementations based on configuration parameters and hardware architecture
- Cannot depend on hybrid modules
- May depend on base modules for concrete implementations
- Provide a unified interface for creating architecture-specific or configuration-specific modules

**Directory Structure:**
```
factory/
├── attention/       # Attention mechanism factories
│   ├── attn_factory.py
│   ├── fmha_impl_base.py
│   ├── cuda_impl/
│   ├── cuda_mla_impl/
│   └── rocm_impl/
├── fused_moe/      # Fused MoE (Mixture of Experts) factories
│   ├── factory.py
│   ├── defs/
│   ├── impl/
│   └── strategy_registry.py
└── linear/         # Linear layer factories
    ├── factory.py
    ├── linear_base.py
    └── impl/
```

**Examples:**
- `LinearFactory`: Creates appropriate linear layer implementations based on quantization, precision, and hardware
- `AttnImplFactory`: Selects attention mechanism implementations (e.g., FlashInfer, TRT, XQA)
- `FusedMoeFactory`: Instantiates Mixture of Experts implementations with various routing and execution strategies

### 3. Hybrid Modules (`hybrid/`)

Hybrid modules are higher-level compositions that assemble base and factory modules into reusable components. These modules are designed for reusing code across different model architectures and are currently architecture-agnostic at this level.

**Characteristics:**
- Compose base and factory modules into cohesive functional units
- Designed for reuse across different model implementations
- Currently no architecture-specific differentiation at this level
- Can depend on both base and factory modules
- Represent common patterns found across multiple model architectures

**Directory Structure:**
```
hybrid/
├── causal_attention.py    # Standard causal attention implementation
├── dense_mlp.py           # Unified dense MLP with multiple activation types (SiGLU, GELU)
└── mla_attention.py       # Multi-head latent attention
```

**Examples:**
- `CausalAttention`: Combines attention factory with normalization and residual connections
- `DenseMLP`: Unified MLP implementation supporting both SiGLU and GELU activations
- `MlaAttention`: Multi-head latent attention mechanism

## Development Guidelines

### Dependency Rules

The module system enforces strict dependency rules to maintain separation of concerns and prevent circular dependencies:

1. **Base Module Dependencies:**
   - ❌ Base modules **CANNOT** depend on factory modules
   - ❌ Base modules **CANNOT** depend on hybrid modules
   - ✅ Base modules may depend on external libraries and utilities

2. **Factory Module Dependencies:**
   - ✅ Factory modules **CAN** depend on base modules
   - ❌ Factory modules **CANNOT** depend on hybrid modules
   - ✅ Factory modules may depend on external libraries and utilities

3. **Hybrid Module Dependencies:**
   - ✅ Hybrid modules **CAN** depend on base modules
   - ✅ Hybrid modules **CAN** depend on factory modules
   - ✅ Hybrid modules may depend on external libraries and utilities

**Dependency Hierarchy:**
```
External Dependencies
        ↓
    Base Modules
        ↓
  Factory Modules
        ↓
  Hybrid Modules
```

### Import Conventions

To maintain clean architecture boundaries, follow these import conventions:

#### 1. Imports Within Modules Directory

When importing between the three subdirectories (base, factory, hybrid), always use the fully qualified path up to the module level:

```python
# ✅ Correct: Import from rtp_llm.models_py.modules.{base|factory|hybrid}
from rtp_llm.models_py.modules.base import RMSNorm, LayerNorm
from rtp_llm.models_py.modules.factory import LinearFactory, AttnImplFactory
from rtp_llm.models_py.modules.hybrid import DenseMLP, CausalAttention
```

```python
# ❌ Incorrect: Don't use imports path within {base|factory|hybrid}
from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm  # Avoid this
```

#### 2. Imports From Outside Modules Directory

When importing from locations outside the `modules` directory, use the parent `modules` level:

```python
# ✅ Correct: Import from rtp_llm.models_py.modules level
from rtp_llm.models_py.modules import (
    RMSNorm,
    LinearFactory,
    DenseMLP,
    CausalAttention,
)
```

This convention ensures:
- Clear visibility of cross-module dependencies
- Easier refactoring and module reorganization
- Consistent import paths across the codebase
- Better adherence to the dependency hierarchy

### Implementation Guidelines

#### Adding a New Base Module

1. Determine if the module needs architecture-specific implementations
2. If architecture-independent, place in `base/common/`
3. If architecture-specific, create implementations in `base/cuda/` and `base/rocm/`
4. Update `base/__init__.py` to export the module with appropriate device type detection
5. Ensure the module has no dependencies on factory or hybrid modules
6. If adding a new base module with only one of architecture-specific, please use `NotImplementedOp` to ensure the import logic

**Example:**
```python
# base/common/my_module.py
import torch
from torch import nn

class MyModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Implementation

    def forward(self, x):
        # Forward logic
        pass
```

#### Adding a New Factory Module

1. Identify the configuration parameters that determine implementation selection
2. Define a base interface or abstract class for the implementations
3. Implement concrete classes for different configurations/architectures
4. Create a factory class with a `create` or similar method
5. Register strategies using the strategy pattern if applicable
6. Update `factory/__init__.py` to export the factory

**Example:**
```python
# factory/my_module/factory.py
from typing import Type, List
from .base import MyModuleBase

class MyModuleFactory:
    _strategies: List[Type[MyModuleBase]] = []

    @classmethod
    def register(cls, strategy_class: Type[MyModuleBase]) -> None:
        cls._strategies.append(strategy_class)

    @classmethod
    def create(cls, config, **kwargs) -> MyModuleBase:
        for strategy in cls._strategies:
            if strategy.can_handle(config):
                return strategy(config, **kwargs)
        raise ValueError("No suitable strategy found")
```

#### Adding a New Hybrid Module

1. Identify the base and factory modules needed for composition
2. Design the interface considering reusability across models
3. Implement the module by composing existing base and factory modules
4. Update `hybrid/__init__.py` to export the module

**Example:**
```python
# hybrid/my_composite.py
from torch import nn
from rtp_llm.models_py.modules.base import RMSNorm
from rtp_llm.models_py.modules.factory import LinearFactory

class MyComposite(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.norm = RMSNorm(config)
        self.linear = LinearFactory.create_linear_from_weights(
            weights, "weight_key", config=config
        )

    def forward(self, x):
        return self.linear(self.norm(x))
```

## Architecture Selection

Architecture-specific implementations are automatically selected based on device type detection:

```python
from rtp_llm.ops.compute_ops import DeviceType, get_device

device_type = get_device().get_device_type()

if device_type == DeviceType.ROCm:
    from rtp_llm.models_py.modules.base.rocm import MyRocmModule
else:
    from rtp_llm.models_py.modules.base.cuda import MyCudaModule
```

This pattern is used throughout the base module `__init__.py` files to provide a unified interface while maintaining architecture-specific optimizations.


## Summary

The three-tier module architecture provides a clean separation between:
- **Base**: Fundamental, architecture-optimized operations
- **Factory**: Smart selection and instantiation logic
- **Hybrid**: Reusable compositions for model building

Following these guidelines ensures maintainable, extensible, and performant model implementations across different hardware architectures and configurations.

