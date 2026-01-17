# Model Configuration Creators

This module provides configuration creation functions that are decoupled from model classes, allowing configuration creation without GPU dependencies.

## Purpose

The `_create_config` methods in model classes have been decoupled to:
- Enable GPU-independent configuration creation (useful for frontend, testing, etc.)
- Improve code organization by separating configuration logic from model implementation
- Allow configuration creation without instantiating model classes

## Usage

### Basic Usage

```python
from rtp_llm.model_config_creators import get_config_creator

# Get configuration creator for a model type
creator = get_config_creator("bert")
if creator:
    config = creator("/path/to/checkpoint")
```

### Using ModelFactory (Recommended)

`ModelFactory.create_model_config()` automatically uses config creators when available:

```python
from rtp_llm.model_factory import ModelFactory
from rtp_llm.config.model_args import ModelArgs

model_args = ModelArgs()
model_args.model_type = "bert"
model_args.ckpt_path = "/path/to/checkpoint"

# This will use config creator if available, otherwise fall back to model class
config = ModelFactory.create_model_config(
    model_args=model_args,
    lora_config=lora_config,
    kv_cache_config=kv_cache_config,
    profiling_debug_logging_config=profiling_debug_logging_config,
)
```

## Migration Guide

To migrate a model's `_create_config` method to a configuration creator:

1. **Create a new file** in `rtp_llm/model_config_creators/` (e.g., `your_model.py`)

2. **Extract the `_create_config` logic** into a function:

```python
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import require_config_json
from rtp_llm.model_config_creators.registry import register_config_creator

def create_your_model_config(ckpt_path: str) -> ModelConfig:
    """Create YourModel configuration from checkpoint path."""
    config = ModelConfig()
    config.ckpt_path = ckpt_path
    # ... extract logic from _create_config ...
    config_json = require_config_json(ckpt_path)
    # ... apply config_json settings ...
    return config

# Register the creator
register_config_creator("your_model", create_your_model_config)
```

3. **Import the module** in `rtp_llm/model_config_creators/__init__.py`:

```python
from rtp_llm.model_config_creators import your_model  # noqa: F401
```

4. **Update the model class** to use the creator (optional, for backward compatibility):

```python
@classmethod
def _create_config(cls, ckpt_path: str) -> ModelConfig:
    """Create model configuration.

    .. deprecated::
        Use rtp_llm.model_config_creators.your_model.create_your_model_config() instead.
    """
    from rtp_llm.model_config_creators.your_model import create_your_model_config
    return create_your_model_config(ckpt_path)
```

## Currently Migrated Models

- `bert`: BERT model configuration
- `roberta`: RoBERTa model configuration
- `tbstars2_5`: TBStars2_5 model configuration

## Testing

Unit tests are located in `rtp_llm/model_config_creators/test/`. Each configuration creator should have tests that verify:
- Correct configuration values are set
- Behavior matches the original `_create_config` method
- Error handling for missing or invalid config.json

## Notes

- Configuration creators should be pure functions (no side effects)
- They should not depend on GPU/CUDA or model class instantiation
- All configuration creators are automatically registered when the module is imported
- ModelFactory will automatically use creators when available, falling back to model classes for backward compatibility

