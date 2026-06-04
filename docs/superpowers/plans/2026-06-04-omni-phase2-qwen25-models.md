# Qwen2.5-Omni Phase 2: Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the three Qwen2.5-Omni stage models (Thinker, Talker, Token2Wav) so that OmniEngine can instantiate each as a separate BaseModel, load weights from the unified HuggingFace checkpoint, and register the `qwen2_5_omni` pipeline topology.

**Architecture:** Each stage is a registered rtp-llm model (`register_model`) with its own Weight class extending `QWenV2Weight` (or `ModelDeployWeightInfo` for Token2Wav). The Weight classes use checkpoint prefixes (`thinker.model.`, `talker.model.`, `token2wav.`) to load from the single unified `.safetensors` checkpoint. `OmniEngine.from_pipeline_config` is enhanced to instantiate each stage's `BaseModel` via `ModelFactory.get_model_cls(stage.model_cls)`, call `from_config`, and wrap in a `Pipeline`. The pipeline topology is registered via `OmniPipelineRegistry.register()` at module import time.

**Tech Stack:** Python 3.10+, PyTorch, existing rtp-llm C++ engine, transformers (for Whisper feature extraction), unittest

## Context

Phase 1 (commits `c19250deb`, `d07b98809`) built the core framework: `OmniPipelineConfig`, `OmniOrchestrator`, `OmniStagePool`, `SharedMemoryConnector`, `OmniOutputProcessor`, `OmniEngine`, pipeline registry, and `ModelFactory` integration. All framework classes exist but are shells — `OmniEngine` creates stage pools without actual `Pipeline` instances, and no real model is registered.

Phase 2 fills in the model layer: actual model classes that the framework instantiates. The HF checkpoint (`Qwen/Qwen2.5-Omni-7B`) stores all component weights in flat sharded safetensors with prefixes: `thinker.model.*`, `thinker.audio_tower.*`, `thinker.lm_head.*`, `talker.model.*`, `talker.codec_head.*`, `talker.thinker_to_talker_proj.*`, `token2wav.dit.*`, `token2wav.bigvgan.*`.

---

## File Structure

```
rtp_llm/omni/models/                          # NEW package
├── __init__.py                                # Package init
├── qwen2_5_omni/
│   ├── __init__.py                            # Imports + PIPELINE registration
│   ├── pipeline_config.py                     # QWEN2_5_OMNI_PIPELINE topology definition
│   ├── thinker.py                             # Qwen2_5OmniThinker model + weight class
│   ├── thinker_audio.py                       # Audio tower (Whisper encoder) for thinker
│   ├── thinker_vision.py                      # Vision encoder for thinker (NaViT)
│   ├── thinker_processor.py                   # Multimodal preprocessor (audio+vision)
│   ├── talker.py                              # Qwen2_5OmniTalker model + weight class
│   ├── token2wav.py                           # Qwen2_5OmniToken2Wav model (DiT + BigVGAN)
│   ├── token2wav_dit.py                       # DiT flow-matching model
│   ├── token2wav_bigvgan.py                   # BigVGAN vocoder
│   └── stage_processors.py                    # thinker2talker, talker2code2wav transforms
rtp_llm/omni/engine/omni_engine.py             # MODIFY — add real stage model instantiation
rtp_llm/omni/__init__.py                       # MODIFY — import models subpackage
rtp_llm/test/omni/                             # Tests
├── test_thinker.py                            # Thinker weight/config tests
├── test_talker.py                             # Talker weight/config tests
├── test_token2wav.py                          # Token2Wav model tests
├── test_stage_processors.py                   # Stage processor tests
├── test_pipeline_config_qwen25.py             # Pipeline topology tests
└── test_omni_engine_integration.py            # OmniEngine real integration test
```

---

### Task 1: Pipeline Topology Registration (`pipeline_config.py`)

**Files:**
- Create: `rtp_llm/omni/models/__init__.py`
- Create: `rtp_llm/omni/models/qwen2_5_omni/__init__.py`
- Create: `rtp_llm/omni/models/qwen2_5_omni/pipeline_config.py`
- Create: `rtp_llm/test/omni/test_pipeline_config_qwen25.py`

This task defines the `QWEN2_5_OMNI_PIPELINE` topology constant and registers it with `OmniPipelineRegistry`. It's the glue that tells the framework "model_type=qwen2_5_omni has 3 stages".

- [ ] **Step 1: Write the failing test for pipeline config**

```python
# rtp_llm/test/omni/test_pipeline_config_qwen25.py
import unittest

from rtp_llm.omni.config.pipeline_registry import OmniPipelineRegistry
from rtp_llm.omni.config.stage_config import StageExecutionType


class TestQwen25OmniPipelineConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import triggers registration
        import rtp_llm.omni.models.qwen2_5_omni  # noqa: F401

    def test_pipeline_registered(self):
        config = OmniPipelineRegistry.get("qwen2_5_omni")
        self.assertIsNotNone(config)
        self.assertEqual(config.model_type, "qwen2_5_omni")
        self.assertEqual(config.model_arch, "Qwen2_5OmniModel")

    def test_pipeline_has_three_stages(self):
        config = OmniPipelineRegistry.get("qwen2_5_omni")
        self.assertEqual(len(config.stages), 3)

    def test_thinker_stage(self):
        config = OmniPipelineRegistry.get("qwen2_5_omni")
        thinker = config.get_stage(0)
        self.assertEqual(thinker.model_stage, "thinker")
        self.assertEqual(thinker.execution_type, StageExecutionType.LLM_AR)
        self.assertEqual(thinker.model_cls, "qwen2_5_omni_thinker")
        self.assertTrue(thinker.final_output)
        self.assertEqual(thinker.final_output_type, "text")
        self.assertTrue(thinker.requires_multimodal_data)

    def test_talker_stage(self):
        config = OmniPipelineRegistry.get("qwen2_5_omni")
        talker = config.get_stage(1)
        self.assertEqual(talker.model_stage, "talker")
        self.assertEqual(talker.execution_type, StageExecutionType.LLM_AR)
        self.assertEqual(talker.model_cls, "qwen2_5_omni_talker")
        self.assertEqual(talker.input_sources, (0,))

    def test_code2wav_stage(self):
        config = OmniPipelineRegistry.get("qwen2_5_omni")
        c2w = config.get_stage(2)
        self.assertEqual(c2w.model_stage, "code2wav")
        self.assertEqual(c2w.execution_type, StageExecutionType.LLM_GENERATION)
        self.assertEqual(c2w.model_cls, "qwen2_5_omni_token2wav")
        self.assertTrue(c2w.final_output)
        self.assertEqual(c2w.final_output_type, "audio")

    def test_final_output_stages(self):
        config = OmniPipelineRegistry.get("qwen2_5_omni")
        finals = config.get_final_output_stages()
        types = {s.final_output_type for s in finals}
        self.assertEqual(types, {"text", "audio"})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/stmatengss/Temp/rtp-llm && python -m pytest rtp_llm/test/omni/test_pipeline_config_qwen25.py -v 2>&1 | head -30`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create package scaffolding and implement pipeline config**

```python
# rtp_llm/omni/models/__init__.py
```

```python
# rtp_llm/omni/models/qwen2_5_omni/pipeline_config.py
from rtp_llm.omni.config.stage_config import (
    OmniPipelineConfig,
    OmniStageConfig,
    StageExecutionType,
)

QWEN2_5_OMNI_PIPELINE = OmniPipelineConfig(
    model_type="qwen2_5_omni",
    model_arch="Qwen2_5OmniModel",
    stages=(
        OmniStageConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            model_cls="qwen2_5_omni_thinker",
            input_sources=(),
            final_output=True,
            final_output_type="text",
            requires_multimodal_data=True,
            engine_output_type="latent",
        ),
        OmniStageConfig(
            stage_id=1,
            model_stage="talker",
            execution_type=StageExecutionType.LLM_AR,
            model_cls="qwen2_5_omni_talker",
            input_sources=(0,),
            engine_output_type="latent",
            stage_processor="qwen2_5_omni.thinker2talker",
        ),
        OmniStageConfig(
            stage_id=2,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            model_cls="qwen2_5_omni_token2wav",
            input_sources=(1,),
            final_output=True,
            final_output_type="audio",
            stage_processor="qwen2_5_omni.talker2code2wav",
        ),
    ),
)
```

```python
# rtp_llm/omni/models/qwen2_5_omni/__init__.py
from rtp_llm.omni.config.pipeline_registry import OmniPipelineRegistry
from rtp_llm.omni.models.qwen2_5_omni.pipeline_config import QWEN2_5_OMNI_PIPELINE

OmniPipelineRegistry.register(QWEN2_5_OMNI_PIPELINE)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/stmatengss/Temp/rtp-llm && python -m pytest rtp_llm/test/omni/test_pipeline_config_qwen25.py -v 2>&1 | head -30`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add rtp_llm/omni/models/__init__.py rtp_llm/omni/models/qwen2_5_omni/__init__.py rtp_llm/omni/models/qwen2_5_omni/pipeline_config.py rtp_llm/test/omni/test_pipeline_config_qwen25.py
git commit -m "feat(omni): add Qwen2.5-Omni pipeline topology registration

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Thinker Model — Weight Class and Model Registration

**Files:**
- Create: `rtp_llm/omni/models/qwen2_5_omni/thinker.py`
- Create: `rtp_llm/test/omni/test_thinker.py`

The thinker's text decoder is architecturally identical to Qwen2.5 (28 layers, GQA, SiLU). Its weights live at `thinker.model.*` and `thinker.lm_head.*` in the checkpoint. We subclass `QWenV2Weight` with `prefix="thinker."` and `QWenV2` with config overrides from `thinker_config.text_config`.

- [ ] **Step 1: Write the failing test for thinker weight and model**

```python
# rtp_llm/test/omni/test_thinker.py
import unittest
import json
import os
import tempfile
from unittest.mock import patch

from rtp_llm.model_factory_register import _model_factory


class TestQwen25OmniThinkerRegistration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import rtp_llm.omni.models.qwen2_5_omni.thinker  # noqa: F401

    def test_model_registered(self):
        self.assertIn("qwen2_5_omni_thinker", _model_factory)

    def test_model_class_name(self):
        model_cls = _model_factory["qwen2_5_omni_thinker"]
        self.assertEqual(model_cls.__name__, "Qwen25OmniThinker")


class TestQwen25OmniThinkerWeight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rtp_llm.omni.models.qwen2_5_omni.thinker import Qwen25OmniThinkerWeight
        cls.weight_cls = Qwen25OmniThinkerWeight

    def test_prefix_is_thinker(self):
        w = self.weight_cls(num_layers=28, hidden_size=3584, head_num=28, head_num_kv=4, size_per_head=128, inter_size=18944)
        self.assertEqual(w.prefix, "thinker.")

    def test_transformer_prefix(self):
        w = self.weight_cls(num_layers=28, hidden_size=3584, head_num=28, head_num_kv=4, size_per_head=128, inter_size=18944)
        w._process_meta([{}], [])
        self.assertEqual(w.transformer_prefix, "thinker.model.")


class TestQwen25OmniThinkerConfig(unittest.TestCase):
    def test_create_config_from_omni_checkpoint(self):
        from rtp_llm.omni.models.qwen2_5_omni.thinker import Qwen25OmniThinker

        # Create a mock config.json matching Qwen2.5-Omni-7B structure
        config_json = {
            "architectures": ["Qwen2_5OmniModel"],
            "model_type": "qwen2_5_omni",
            "thinker_config": {
                "text_config": {
                    "hidden_size": 3584,
                    "intermediate_size": 18944,
                    "num_attention_heads": 28,
                    "num_key_value_heads": 4,
                    "num_hidden_layers": 28,
                    "vocab_size": 152064,
                    "rms_norm_eps": 1e-06,
                    "rope_theta": 1000000.0,
                    "tie_word_embeddings": False,
                    "torch_dtype": "bfloat16",
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config_json, f)

            config = Qwen25OmniThinker._create_config(tmpdir)
            self.assertEqual(config.hidden_size, 3584)
            self.assertEqual(config.num_layers, 28)
            self.assertEqual(config.attn_config.head_num, 28)
            self.assertEqual(config.attn_config.kv_head_num, 4)
            self.assertEqual(config.inter_size, 18944)
            self.assertEqual(config.vocab_size, 152064)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/stmatengss/Temp/rtp-llm && python -m pytest rtp_llm/test/omni/test_thinker.py -v 2>&1 | head -30`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement Thinker model and weight class**

```python
# rtp_llm/omni/models/qwen2_5_omni/thinker.py
import json
import os
from typing import Any, Dict, List

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.qwen_v2 import QWenV2, QWenV2Weight
from rtp_llm.utils.util import get_config_from_path


class Qwen25OmniThinkerWeight(QWenV2Weight):
    def __init__(self, **kwargs: Any):
        super().__init__(prefix="thinker.", **kwargs)


class Qwen25OmniThinker(QWenV2):
    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.max_seq_len = 32768
        config.attn_config.rope_config.dim = 128
        config.attn_config.rope_config.style = 1
        config.has_pre_decoder_layernorm = False
        config.special_tokens.bos_token_id = 151644
        config.special_tokens.eos_token_id = 151645

        config_json = get_config_from_path(ckpt_path)
        if config_json and "thinker_config" in config_json:
            text_config = config_json["thinker_config"].get("text_config", {})
            QWenV2._from_config_json(config, text_config)
        else:
            cls._from_hf(config, ckpt_path)
        return config

    @staticmethod
    def get_weight_cls():
        return Qwen25OmniThinkerWeight


register_model("qwen2_5_omni_thinker", Qwen25OmniThinker)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/stmatengss/Temp/rtp-llm && python -m pytest rtp_llm/test/omni/test_thinker.py -v 2>&1 | head -40`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add rtp_llm/omni/models/qwen2_5_omni/thinker.py rtp_llm/test/omni/test_thinker.py
git commit -m "feat(omni): add Qwen2.5-Omni thinker stage model and weight class

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Thinker Multimodal — Audio Tower (Whisper Encoder)

**Files:**
- Create: `rtp_llm/omni/models/qwen2_5_omni/thinker_audio.py`
- Modify: `rtp_llm/omni/models/qwen2_5_omni/thinker.py` (add multimodal mixin)

The thinker has a Whisper-style audio encoder at `thinker.audio_tower.*` (32 layers, d_model=1280, 20 heads, 128 mel bins). This is similar to the existing `rtp_llm/models/qwen_v2_audio/` but with different weight prefixes and config structure.

The existing `Qwen2AudioEncoder` in `rtp_llm/models/qwen_v2_audio/modeling_qwen2_audio.py` can be reused — it's the same Whisper architecture. We just need a new `Processor` that reads from the omni config structure (`thinker_config.audio_config`) instead of `audio_config`.

- [ ] **Step 1: Write the failing test for thinker audio processor**

```python
# Add to rtp_llm/test/omni/test_thinker.py

class TestQwen25OmniThinkerAudioProcessor(unittest.TestCase):
    def test_processor_class_exists(self):
        from rtp_llm.omni.models.qwen2_5_omni.thinker_audio import OmniAudioProcessor
        self.assertTrue(callable(OmniAudioProcessor))

    def test_processor_reads_omni_audio_config(self):
        from rtp_llm.omni.models.qwen2_5_omni.thinker_audio import OmniAudioProcessor
        # Verify it accepts audio_config dict format from thinker_config
        audio_config = {
            "d_model": 1280,
            "encoder_attention_heads": 20,
            "encoder_ffn_dim": 5120,
            "encoder_layers": 32,
            "num_mel_bins": 128,
            "output_dim": 3584,
            "max_source_positions": 1500,
            "n_window": 100,
        }
        # Should not raise
        OmniAudioProcessor.validate_config(audio_config)
```

- [ ] **Step 2: Implement OmniAudioProcessor**

```python
# rtp_llm/omni/models/qwen2_5_omni/thinker_audio.py
import torch
from typing import Dict

from rtp_llm.models.qwen_v2_audio.modeling_qwen2_audio import Qwen2AudioEncoder
from rtp_llm.models.qwen_v2_audio.configuration_qwen2_audio import Qwen2AudioEncoderConfig


class OmniAudioProcessor:
    """Audio encoder for Qwen2.5-Omni thinker stage.

    Wraps the same Whisper-based encoder used by qwen_v2_audio, but
    reads config from thinker_config.audio_config format.
    """

    def __init__(self, audio_config_dict: Dict):
        encoder_config = Qwen2AudioEncoderConfig.from_dict(audio_config_dict)
        self.audio_tower = Qwen2AudioEncoder._from_config(encoder_config)

    @staticmethod
    def validate_config(audio_config: Dict) -> None:
        required = ["d_model", "encoder_attention_heads", "encoder_layers", "num_mel_bins"]
        for key in required:
            if key not in audio_config:
                raise ValueError(f"Missing required audio config key: {key}")

    @property
    def device(self):
        return next(self.audio_tower.parameters()).device

    @torch.inference_mode()
    def encode(self, input_features: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        return self.audio_tower(input_features, attention_mask=attention_mask)
```

- [ ] **Step 3: Run tests to verify they pass**
- [ ] **Step 4: Commit**

---

### Task 4: Thinker Multimodal — Vision Encoder (NaViT)

**Files:**
- Create: `rtp_llm/omni/models/qwen2_5_omni/thinker_vision.py`

The vision encoder uses the same architecture as Qwen2.5-VL (`rtp_llm/models/qwen2_5_vl/`). Weights are at `thinker.visual.*`. Reuse `rtp_llm/models/qwen2_5_vl/modeling_qwen2_5_vl.py` for the ViT model.

- [ ] **Step 1: Write the failing test**

```python
# rtp_llm/test/omni/test_thinker.py — append

class TestQwen25OmniVisionEncoder(unittest.TestCase):
    def test_vision_processor_exists(self):
        from rtp_llm.omni.models.qwen2_5_omni.thinker_vision import OmniVisionProcessor
        self.assertTrue(callable(OmniVisionProcessor))

    def test_vision_config_from_omni(self):
        from rtp_llm.omni.models.qwen2_5_omni.thinker_vision import OmniVisionProcessor
        vision_config = {
            "depth": 32,
            "embed_dim": 1280,
            "hidden_size": 1280,
            "num_heads": 16,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "out_hidden_size": 3584,
        }
        OmniVisionProcessor.validate_config(vision_config)
```

- [ ] **Step 2: Implement OmniVisionProcessor**

```python
# rtp_llm/omni/models/qwen2_5_omni/thinker_vision.py
import torch
from typing import Dict


class OmniVisionProcessor:
    """Vision encoder for Qwen2.5-Omni thinker stage.

    Reuses Qwen2.5-VL's ViT with NaViT patches.
    """

    def __init__(self, vision_config_dict: Dict):
        self.config = vision_config_dict
        # Lazy init — actual ViT model loaded during weight loading

    @staticmethod
    def validate_config(vision_config: Dict) -> None:
        required = ["depth", "embed_dim", "num_heads", "patch_size"]
        for key in required:
            if key not in vision_config:
                raise ValueError(f"Missing required vision config key: {key}")

    @torch.inference_mode()
    def encode(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor = None) -> torch.Tensor:
        return self.vit(pixel_values, grid_thw=grid_thw)
```

- [ ] **Step 3: Run tests, commit**

---

### Task 5: Thinker Multimodal Processor Integration

**Files:**
- Create: `rtp_llm/omni/models/qwen2_5_omni/thinker_processor.py`
- Modify: `rtp_llm/omni/models/qwen2_5_omni/thinker.py` — add MultiModalMixin

Integrates audio + vision processors into the thinker model using the existing `MultiModalMixin` pattern (same as `QWenV2Audio`).

- [ ] **Step 1: Write the failing test**

```python
# rtp_llm/test/omni/test_thinker.py — append

class TestQwen25OmniThinkerMultimodal(unittest.TestCase):
    def test_thinker_has_multimodal_mixin(self):
        from rtp_llm.omni.models.qwen2_5_omni.thinker import Qwen25OmniThinker
        from rtp_llm.models.multimodal.multimodal_mixin import MultiModalMixin
        self.assertTrue(issubclass(Qwen25OmniThinker, MultiModalMixin))
```

- [ ] **Step 2: Implement thinker processor and modify thinker model**

```python
# rtp_llm/omni/models/qwen2_5_omni/thinker_processor.py
from io import BytesIO
from typing import Dict

import torch
from rtp_llm.config.model_config import VitParameters
from rtp_llm.models.multimodal.multimodal_common import AudioEmbeddingInterface
from rtp_llm.omni.models.qwen2_5_omni.thinker_audio import OmniAudioProcessor
from rtp_llm.utils.util import get_config_from_path


class OmniThinkerProcessor(AudioEmbeddingInterface):
    def __init__(self, mm_related_params: VitParameters, ckpt_path: str):
        self.mm_related_params = mm_related_params
        config_json = get_config_from_path(ckpt_path)
        thinker_cfg = config_json.get("thinker_config", {})

        audio_config = thinker_cfg.get("audio_config", {})
        if audio_config:
            self.audio_processor = OmniAudioProcessor(audio_config)
        else:
            self.audio_processor = None

    @property
    def _device(self):
        if self.audio_processor:
            return self.audio_processor.device
        return torch.device("cpu")

    @torch.inference_mode()
    def audio_embedding(self, features_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_features = features_dict["input_features"].to(self._device)
        return self.audio_processor.encode(input_features)
```

Then modify `thinker.py` to add `MultiModalMixin` inheritance (same pattern as `QWenV2Audio`):

```python
# Modify Qwen25OmniThinker to extend both QWenV2 and MultiModalMixin
class Qwen25OmniThinker(QWenV2, MultiModalMixin):
    def _init_multimodal(self, mm_model_config, vit_config):
        self.mm_part = OmniThinkerProcessor(
            self.model_config.mm_related_params,
            self.model_config.ckpt_path,
        )
        # ... set vit_weights for weight loading
```

Weight class also needs `BaseMultiModalWeightInfo` mixin:

```python
class Qwen25OmniThinkerWeight(QWenV2Weight, BaseMultiModalWeightInfo):
    def __init__(self, vit_weights=None, **kwargs):
        QWenV2Weight.__init__(self, prefix="thinker.", **kwargs)
        BaseMultiModalWeightInfo.__init__(self, vit_weights=vit_weights, **kwargs)
```

- [ ] **Step 3: Run tests, commit**

---

### Task 6: Talker Model — Weight Class and Model Registration

**Files:**
- Create: `rtp_llm/omni/models/qwen2_5_omni/talker.py`
- Create: `rtp_llm/test/omni/test_talker.py`

The talker is a smaller Qwen2 model (24 layers, hidden=896, 12 heads, 4 KV heads, vocab=8448). Weights at `talker.model.*`, `talker.codec_head.*`, `talker.thinker_to_talker_proj.*`.

Key differences from standard QWenV2:
- `prefix = "talker."` for all weights
- `codec_head` instead of `lm_head` for the output projection
- `thinker_to_talker_proj` as an additional weight (cross-model projection)
- Different vocab size (8448 for codec tokens)
- `embedding_size=3584` (thinker hidden dim) for the cross-attention projection

- [ ] **Step 1: Write the failing test**

```python
# rtp_llm/test/omni/test_talker.py
import unittest
import json
import os
import tempfile

from rtp_llm.model_factory_register import _model_factory


class TestQwen25OmniTalkerRegistration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import rtp_llm.omni.models.qwen2_5_omni.talker  # noqa: F401

    def test_model_registered(self):
        self.assertIn("qwen2_5_omni_talker", _model_factory)


class TestQwen25OmniTalkerWeight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rtp_llm.omni.models.qwen2_5_omni.talker import Qwen25OmniTalkerWeight
        cls.weight_cls = Qwen25OmniTalkerWeight

    def test_prefix_is_talker(self):
        w = self.weight_cls(num_layers=24, hidden_size=896, head_num=12, head_num_kv=4, size_per_head=128, inter_size=18944)
        self.assertEqual(w.prefix, "talker.")

    def test_transformer_prefix(self):
        w = self.weight_cls(num_layers=24, hidden_size=896, head_num=12, head_num_kv=4, size_per_head=128, inter_size=18944)
        w._process_meta([{}], [])
        self.assertEqual(w.transformer_prefix, "talker.model.")


class TestQwen25OmniTalkerConfig(unittest.TestCase):
    def test_create_config_from_omni_checkpoint(self):
        from rtp_llm.omni.models.qwen2_5_omni.talker import Qwen25OmniTalker

        config_json = {
            "architectures": ["Qwen2_5OmniModel"],
            "model_type": "qwen2_5_omni",
            "talker_config": {
                "hidden_size": 896,
                "intermediate_size": 18944,
                "num_attention_heads": 12,
                "num_key_value_heads": 4,
                "num_hidden_layers": 24,
                "vocab_size": 8448,
                "rms_norm_eps": 1e-06,
                "rope_theta": 1000000.0,
                "embedding_size": 3584,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config_json, f)

            config = Qwen25OmniTalker._create_config(tmpdir)
            self.assertEqual(config.hidden_size, 896)
            self.assertEqual(config.num_layers, 24)
            self.assertEqual(config.attn_config.head_num, 12)
            self.assertEqual(config.attn_config.kv_head_num, 4)
            self.assertEqual(config.vocab_size, 8448)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/stmatengss/Temp/rtp-llm && python -m pytest rtp_llm/test/omni/test_talker.py -v 2>&1 | head -30`
Expected: FAIL

- [ ] **Step 3: Implement Talker model and weight class**

```python
# rtp_llm/omni/models/qwen2_5_omni/talker.py
import functools
from typing import Any, List

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.qwen_v2 import QWenV2, QWenV2Weight
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity
from rtp_llm.utils.util import get_config_from_path


class Qwen25OmniTalkerWeight(QWenV2Weight):
    def __init__(self, **kwargs: Any):
        super().__init__(prefix="talker.", **kwargs)

    def _get_hf_weight_info(self):
        weight_info = super()._get_hf_weight_info()
        # Override lm_head to use codec_head
        for i, w in enumerate(weight_info.weights):
            if w.name == W.lm_head:
                weight_info.weights[i] = AtomicWeight(
                    W.lm_head,
                    [CkptWeightInfo("talker.codec_head.weight", identity)],
                    identity,
                )
                break
        # Add thinker_to_talker_proj as misc weight
        weight_info.weights.append(
            AtomicWeight(
                "thinker_to_talker_proj_weight",
                [CkptWeightInfo("talker.thinker_to_talker_proj.weight", identity)],
                identity,
            )
        )
        weight_info.weights.append(
            AtomicWeight(
                "thinker_to_talker_proj_bias",
                [CkptWeightInfo("talker.thinker_to_talker_proj.bias", identity)],
                identity,
            )
        )
        return weight_info


class Qwen25OmniTalker(QWenV2):
    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.max_seq_len = 32768
        config.attn_config.rope_config.dim = 128
        config.attn_config.rope_config.style = 1
        config.has_pre_decoder_layernorm = False

        config_json = get_config_from_path(ckpt_path)
        if config_json and "talker_config" in config_json:
            talker_cfg = config_json["talker_config"]
            QWenV2._from_config_json(config, talker_cfg)
        else:
            cls._from_hf(config, ckpt_path)
        return config

    @staticmethod
    def get_weight_cls():
        return Qwen25OmniTalkerWeight


register_model("qwen2_5_omni_talker", Qwen25OmniTalker)
```

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

---

### Task 7: Token2Wav — DiT Flow-Matching Model

**Files:**
- Create: `rtp_llm/omni/models/qwen2_5_omni/token2wav_dit.py`
- Create: `rtp_llm/test/omni/test_token2wav.py`

The DiT (Diffusion Transformer) is a flow-matching model: 22 transformer layers, dim=1024, 16 heads, head_dim=64. It takes codec embeddings + speaker embeddings and produces mel spectrograms. This is a completely new architecture — no existing rtp-llm model to reuse.

Checkpoint weights: `token2wav.dit.*`

The DiT has these sub-components:
- Codec embedding: `num_embeds=8193`, `emb_dim=512`
- Speaker encoder: ECAPA-TDNN style (enc_channels=[256,256,256,256,768])
- Transformer blocks: 22 layers with AdaLN-Zero conditioning
- Mel projection head: projects to `mel_dim=80`

- [ ] **Step 1: Write the failing test**

```python
# rtp_llm/test/omni/test_token2wav.py
import unittest
import torch


class TestOmniDiT(unittest.TestCase):
    def test_dit_model_class_exists(self):
        from rtp_llm.omni.models.qwen2_5_omni.token2wav_dit import OmniDiT
        self.assertTrue(callable(OmniDiT))

    def test_dit_config_from_dict(self):
        from rtp_llm.omni.models.qwen2_5_omni.token2wav_dit import OmniDiTConfig
        config = OmniDiTConfig.from_dict({
            "depth": 22, "dim": 1024, "heads": 16, "head_dim": 64,
            "ff_mult": 2, "mel_dim": 80, "num_embeds": 8193,
            "emb_dim": 512, "dropout": 0.1,
        })
        self.assertEqual(config.depth, 22)
        self.assertEqual(config.dim, 1024)

    def test_dit_forward_shape(self):
        from rtp_llm.omni.models.qwen2_5_omni.token2wav_dit import OmniDiT, OmniDiTConfig
        config = OmniDiTConfig(
            depth=2, dim=64, heads=4, head_dim=16,
            ff_mult=2, mel_dim=80, num_embeds=100, emb_dim=32,
            dropout=0.0,
        )
        model = OmniDiT(config)
        # batch=1, seq_len=10, mel_dim=80
        x = torch.randn(1, 10, 80)
        t = torch.tensor([0.5])
        codec_ids = torch.randint(0, 100, (1, 10))
        out = model(x, t, codec_ids)
        self.assertEqual(out.shape, (1, 10, 80))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Implement OmniDiT**

The DiT architecture follows the flow-matching pattern:
- Sinusoidal time embedding
- Codec token embedding
- N transformer blocks with AdaLN-Zero
- Final layer norm + projection to mel dim

```python
# rtp_llm/omni/models/qwen2_5_omni/token2wav_dit.py
import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OmniDiTConfig:
    depth: int = 22
    dim: int = 1024
    heads: int = 16
    head_dim: int = 64
    ff_mult: int = 2
    mel_dim: int = 80
    num_embeds: int = 8193
    emb_dim: int = 512
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, d: Dict) -> "OmniDiTConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, ff_mult: int, dropout: float):
        super().__init__()
        inner_dim = heads * head_dim
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )
        # AdaLN-Zero: 6 parameters per block (gamma1, beta1, alpha1, gamma2, beta2, alpha2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        mod = self.adaLN_modulation(cond).unsqueeze(1)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)

        h = self.norm1(x) * (1 + gamma1) + beta1
        h, _ = self.attn(h, h, h)
        x = x + alpha1 * h

        h = self.norm2(x) * (1 + gamma2) + beta2
        h = self.ff(h)
        x = x + alpha2 * h
        return x


class OmniDiT(nn.Module):
    def __init__(self, config: OmniDiTConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.mel_dim, config.dim)
        self.codec_embed = nn.Embedding(config.num_embeds, config.emb_dim)
        self.codec_proj = nn.Linear(config.emb_dim, config.dim)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(config.dim),
            nn.Linear(config.dim, config.dim),
            nn.SiLU(),
            nn.Linear(config.dim, config.dim),
        )
        self.blocks = nn.ModuleList([
            DiTBlock(config.dim, config.heads, config.head_dim, config.ff_mult, config.dropout)
            for _ in range(config.depth)
        ])
        self.final_norm = nn.LayerNorm(config.dim)
        self.output_proj = nn.Linear(config.dim, config.mel_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, codec_ids: torch.Tensor,
                spk_emb: torch.Tensor = None) -> torch.Tensor:
        # x: (B, T, mel_dim), t: (B,), codec_ids: (B, T)
        h = self.input_proj(x)
        codec_h = self.codec_proj(self.codec_embed(codec_ids))
        h = h + codec_h
        t_emb = self.time_embed(t)

        for block in self.blocks:
            h = block(h, t_emb)

        h = self.final_norm(h)
        return self.output_proj(h)
```

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

---

### Task 8: Token2Wav — BigVGAN Vocoder

**Files:**
- Create: `rtp_llm/omni/models/qwen2_5_omni/token2wav_bigvgan.py`
- Modify: `rtp_llm/test/omni/test_token2wav.py` (append tests)

BigVGAN converts mel spectrograms to raw audio waveforms. Checkpoint weights: `token2wav.bigvgan.*`.

Config from HF: upsample_rates=[5,3,2,2,2,2] (total 240x), upsample_initial_channel=1536, resblock_kernel_sizes=[3,7,11], resblock_dilation_sizes=[[1,3,5],[1,3,5],[1,3,5]], mel_dim=80.

- [ ] **Step 1: Write the failing test**

```python
# Append to rtp_llm/test/omni/test_token2wav.py

class TestBigVGAN(unittest.TestCase):
    def test_bigvgan_class_exists(self):
        from rtp_llm.omni.models.qwen2_5_omni.token2wav_bigvgan import BigVGAN
        self.assertTrue(callable(BigVGAN))

    def test_bigvgan_forward_shape(self):
        from rtp_llm.omni.models.qwen2_5_omni.token2wav_bigvgan import BigVGAN, BigVGANConfig
        config = BigVGANConfig(
            mel_dim=80,
            upsample_rates=[2, 2],
            upsample_initial_channel=64,
            upsample_kernel_sizes=[4, 4],
            resblock_kernel_sizes=[3],
            resblock_dilation_sizes=[[1, 2]],
        )
        model = BigVGAN(config)
        # (B, mel_dim, T)
        mel = torch.randn(1, 80, 10)
        wav = model(mel)
        # upsample 2*2 = 4x
        self.assertEqual(wav.shape[0], 1)
        self.assertEqual(wav.shape[1], 1)
        self.assertEqual(wav.shape[2], 40)
```

- [ ] **Step 2: Implement BigVGAN**

Standard HiFi-GAN / BigVGAN architecture with multi-receptive-field fusion (MRF) residual blocks and transposed convolution upsampling.

- [ ] **Step 3: Run tests, commit**

---

### Task 9: Token2Wav — Combined Model Registration

**Files:**
- Create: `rtp_llm/omni/models/qwen2_5_omni/token2wav.py`
- Modify: `rtp_llm/test/omni/test_token2wav.py` (append tests)

Combines DiT + BigVGAN into a single `Qwen25OmniToken2Wav` model class that is registered as `qwen2_5_omni_token2wav`. The model takes codec token IDs → DiT generates mel → BigVGAN generates waveform.

- [ ] **Step 1: Write the failing test**

```python
class TestQwen25OmniToken2WavRegistration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import rtp_llm.omni.models.qwen2_5_omni.token2wav  # noqa: F401

    def test_model_registered(self):
        self.assertIn("qwen2_5_omni_token2wav", _model_factory)


class TestQwen25OmniToken2WavConfig(unittest.TestCase):
    def test_create_config_from_omni_checkpoint(self):
        from rtp_llm.omni.models.qwen2_5_omni.token2wav import Qwen25OmniToken2Wav
        config_json = {
            "token2wav_config": {
                "dit_config": {
                    "depth": 22, "dim": 1024, "heads": 16, "head_dim": 64,
                    "ff_mult": 2, "mel_dim": 80, "num_embeds": 8193, "emb_dim": 512,
                },
                "bigvgan_config": {
                    "mel_dim": 80,
                    "upsample_rates": [5, 3, 2, 2, 2, 2],
                    "upsample_initial_channel": 1536,
                    "upsample_kernel_sizes": [11, 7, 4, 4, 4, 4],
                    "resblock_kernel_sizes": [3, 7, 11],
                    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config_json, f)
            config = Qwen25OmniToken2Wav._create_config(tmpdir)
            self.assertIsNotNone(config)
```

- [ ] **Step 2: Implement Token2Wav model**

The `Qwen25OmniToken2Wav` does NOT extend `BaseModel` in the standard way since it's a non-AR model. Instead it registers as a standalone model class that the `OmniStagePool` invokes directly (similar to `StageExecutionType.LLM_GENERATION`). Its weight class (`Qwen25OmniToken2WavWeight`) extends `ModelDeployWeightInfo` directly with custom weight mapping for `token2wav.dit.*` and `token2wav.bigvgan.*`.

- [ ] **Step 3: Run tests, commit**

---

### Task 10: Stage Processors — thinker2talker and talker2code2wav

**Files:**
- Create: `rtp_llm/omni/models/qwen2_5_omni/stage_processors.py`
- Create: `rtp_llm/test/omni/test_stage_processors.py`

Stage processors transform `StageOutput` between stages:

1. **thinker2talker**: Extracts hidden state embeddings from thinker output → formats as talker input (applies `thinker_to_talker_proj`)
2. **talker2code2wav**: Converts talker's codec token IDs → Token2Wav input format

Both extend `StageProcessorBase` from Phase 1.

- [ ] **Step 1: Write the failing test**

```python
# rtp_llm/test/omni/test_stage_processors.py
import unittest
import torch

from rtp_llm.omni.engine.stage_connector import StageOutput


class TestThinker2TalkerProcessor(unittest.TestCase):
    def test_processor_exists(self):
        from rtp_llm.omni.models.qwen2_5_omni.stage_processors import Thinker2TalkerProcessor
        self.assertTrue(callable(Thinker2TalkerProcessor))

    def test_process_extracts_embeddings(self):
        from rtp_llm.omni.models.qwen2_5_omni.stage_processors import Thinker2TalkerProcessor
        proc = Thinker2TalkerProcessor()
        thinker_output = StageOutput(
            token_ids=[1, 2, 3],
            embeddings=torch.randn(1, 10, 3584),
            metadata={"text": "hello"},
        )
        talker_input = proc.process(thinker_output)
        self.assertIsNotNone(talker_input.embeddings)
        self.assertEqual(talker_input.embeddings.shape[-1], 3584)


class TestTalker2Code2WavProcessor(unittest.TestCase):
    def test_processor_exists(self):
        from rtp_llm.omni.models.qwen2_5_omni.stage_processors import Talker2Code2WavProcessor
        self.assertTrue(callable(Talker2Code2WavProcessor))

    def test_process_converts_tokens(self):
        from rtp_llm.omni.models.qwen2_5_omni.stage_processors import Talker2Code2WavProcessor
        proc = Talker2Code2WavProcessor()
        talker_output = StageOutput(
            token_ids=[10, 20, 30, 40, 50],
            metadata={"codec_tokens": True},
        )
        c2w_input = proc.process(talker_output)
        self.assertIsNotNone(c2w_input.token_ids)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Implement stage processors**

```python
# rtp_llm/omni/models/qwen2_5_omni/stage_processors.py
import torch
from rtp_llm.omni.engine.stage_connector import StageOutput
from rtp_llm.omni.engine.stage_processor_base import StageProcessorBase


class Thinker2TalkerProcessor(StageProcessorBase):
    def process(self, source_output: StageOutput) -> StageOutput:
        return StageOutput(
            embeddings=source_output.embeddings,
            metadata={
                "source_token_ids": source_output.token_ids,
                "source_text": source_output.metadata.get("text", ""),
            },
        )


class Talker2Code2WavProcessor(StageProcessorBase):
    def process(self, source_output: StageOutput) -> StageOutput:
        return StageOutput(
            token_ids=source_output.token_ids,
            metadata={"from_talker": True},
        )
```

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

---

### Task 11: OmniEngine Enhancement — Real Stage Instantiation

**Files:**
- Modify: `rtp_llm/omni/engine/omni_engine.py`
- Create: `rtp_llm/test/omni/test_omni_engine_integration.py`

Currently `OmniEngine.__init__` creates `OmniStagePool` instances without actual `Pipeline` replicas. This task enhances `OmniEngine.from_pipeline_config` to:
1. For each stage, look up the model class via `ModelFactory.get_model_cls(stage.model_cls)`
2. Create a per-stage `ModelConfig` by calling `model_cls._create_config(ckpt_path)`
3. Call `model_cls.from_config(...)` to instantiate the model
4. Create a `Pipeline` instance wrapping the C++ engine
5. Add the pipeline as a replica in `OmniStagePool`

- [ ] **Step 1: Write the failing test**

```python
# rtp_llm/test/omni/test_omni_engine_integration.py
import unittest
from unittest.mock import MagicMock, patch

from rtp_llm.omni.config.pipeline_registry import OmniPipelineRegistry


class TestOmniEngineStageInstantiation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import rtp_llm.omni.models.qwen2_5_omni  # noqa: F401

    def test_engine_resolves_stage_model_classes(self):
        from rtp_llm.omni.engine.omni_engine import OmniEngine
        from rtp_llm.model_factory_register import _model_factory

        config = OmniPipelineRegistry.get("qwen2_5_omni")
        self.assertIsNotNone(config)

        for stage in config.stages:
            self.assertIn(stage.model_cls, _model_factory,
                f"Stage model_cls '{stage.model_cls}' not registered")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Enhance OmniEngine.from_pipeline_config**

Add `_init_stage_models` method that iterates over stages and resolves model classes. The actual model instantiation (weight loading, Pipeline creation) requires a real GPU and checkpoint, so we gate it behind a `lazy_init` flag — on init, we validate that all stage model classes are registered; actual Pipeline creation happens in a `start()` call.

```python
# In omni_engine.py, add:
from rtp_llm.model_factory_register import _model_factory

class OmniEngine:
    def _validate_stage_models(self) -> None:
        for stage in self.pipeline_config.stages:
            if stage.model_cls not in _model_factory:
                raise ValueError(
                    f"Stage model '{stage.model_cls}' not registered. "
                    f"Import the model module first."
                )

    @classmethod
    def from_pipeline_config(cls, pipeline_config, model_config=None, engine_config=None):
        engine = cls(
            pipeline_config=pipeline_config,
            model_config=model_config,
            engine_config=engine_config,
        )
        engine._validate_stage_models()
        return engine
```

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

---

### Task 12: Module Imports and Package Wiring

**Files:**
- Modify: `rtp_llm/omni/__init__.py` — import models subpackage
- Modify: `rtp_llm/omni/models/qwen2_5_omni/__init__.py` — import all stage models

Ensure that importing `rtp_llm.omni` transitively registers all Qwen2.5-Omni stage models and the pipeline topology.

- [ ] **Step 1: Write the failing test**

```python
# rtp_llm/test/omni/test_omni_engine_integration.py — append

class TestOmniImportChain(unittest.TestCase):
    def test_importing_omni_registers_pipeline(self):
        import rtp_llm.omni  # noqa: F401
        config = OmniPipelineRegistry.get("qwen2_5_omni")
        self.assertIsNotNone(config)

    def test_importing_omni_registers_all_stage_models(self):
        import rtp_llm.omni  # noqa: F401
        from rtp_llm.model_factory_register import _model_factory
        for model_name in ["qwen2_5_omni_thinker", "qwen2_5_omni_talker", "qwen2_5_omni_token2wav"]:
            self.assertIn(model_name, _model_factory, f"{model_name} not registered")
```

- [ ] **Step 2: Update imports**

```python
# rtp_llm/omni/__init__.py — add at end:
import rtp_llm.omni.models  # noqa: F401
```

```python
# rtp_llm/omni/models/qwen2_5_omni/__init__.py — add imports:
from rtp_llm.omni.config.pipeline_registry import OmniPipelineRegistry
from rtp_llm.omni.models.qwen2_5_omni.pipeline_config import QWEN2_5_OMNI_PIPELINE
import rtp_llm.omni.models.qwen2_5_omni.thinker  # noqa: F401
import rtp_llm.omni.models.qwen2_5_omni.talker  # noqa: F401
import rtp_llm.omni.models.qwen2_5_omni.token2wav  # noqa: F401

OmniPipelineRegistry.register(QWEN2_5_OMNI_PIPELINE)
```

- [ ] **Step 3: Run all omni tests**

Run: `cd /Users/stmatengss/Temp/rtp-llm && python -m pytest rtp_llm/test/omni/ -v 2>&1 | tail -30`
Expected: All PASS (including existing Phase 1 tests)

- [ ] **Step 4: Commit**

---

### Task 13: Run Full Test Suite and Verify No Regressions

**Files:** None (verification only)

- [ ] **Step 1: Run all omni tests**

Run on remote dev server:
```bash
ssh root@mateng04 "cd /path/to/rtp-llm && python -m pytest rtp_llm/test/omni/ -v 2>&1"
```

- [ ] **Step 2: Run existing model tests to verify no regressions**

```bash
ssh root@mateng04 "cd /path/to/rtp-llm && python -m pytest rtp_llm/test/ -k 'not gpu and not cuda' --timeout=120 -v 2>&1 | tail -50"
```

- [ ] **Step 3: Verify import chain doesn't break existing models**

```bash
python -c "import rtp_llm.models; import rtp_llm.omni; print('OK')"
```

---

## Key Design Decisions

1. **Weight prefix pattern**: Each stage's Weight class extends `QWenV2Weight` with a different `prefix` (`"thinker."`, `"talker."`). This leverages the existing `prefix + model_prefix + "layers.{i}..."` pattern — no changes to the weight loading infrastructure.

2. **Token2Wav is non-standard**: It doesn't follow the AR decoder pattern, so it extends `ModelDeployWeightInfo` directly rather than `QWenV2Weight`. The DiT and BigVGAN are pure PyTorch `nn.Module`s.

3. **Lazy stage initialization**: `OmniEngine.from_pipeline_config` validates model registrations but doesn't load weights or create Pipelines — that requires GPU access and happens at engine start time. This keeps the cold path fast and testable without GPU.

4. **Reuse existing encoders**: The audio tower reuses `Qwen2AudioEncoder` from `rtp_llm/models/qwen_v2_audio/`. The vision encoder reuses patterns from `rtp_llm/models/qwen2_5_vl/`.

## Verification

1. **Unit tests**: `python -m pytest rtp_llm/test/omni/ -v` — all pass
2. **Import smoke test**: `python -c "import rtp_llm.omni; print('OK')"` — no import errors
3. **Weight prefix verification** (requires checkpoint): Load `Qwen/Qwen2.5-Omni-7B` and verify weight keys match prefixes
4. **E2E test** (requires GPU): Full pipeline inference with text+audio output — this is Phase 2's final validation, likely deferred to after all tasks complete and done on remote dev server (`ssh root@mateng04`)
