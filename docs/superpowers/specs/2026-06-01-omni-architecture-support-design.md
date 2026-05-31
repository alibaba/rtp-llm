# Omni Architecture Support for rtp-llm

**Date**: 2026-06-01
**Status**: Draft
**Target Models**: Qwen2.5-Omni, Qwen3-Omni (initial), extensible to all vllm-omni models

## 1. Problem Statement

rtp-llm currently supports multimodal **input** (vision, audio understanding) through the `MultiModalMixin` pattern, where encoders (ViT, Whisper) preprocess inputs before the LLM decoder. However, it has no support for multimodal **output** — no TTS/speech generation, no image generation, no video generation.

Modern "omni" models (Qwen2.5-Omni, Qwen3-Omni, Bagel, GLM-Image) require a **multi-stage pipeline** architecture: an LLM "thinker" generates text + hidden embeddings, a "talker" converts embeddings to speech tokens, and a "code2wav" module converts speech tokens to audio waveforms. Image/video generation models use diffusion transformers (DiT) as a pipeline stage.

This design adds omni architecture support to rtp-llm by porting vllm-omni's multi-stage pipeline framework, enabling rtp-llm to serve models that produce text, audio, image, and video outputs.

## 2. Architecture Overview

### 2.1 Multi-Stage Pipeline Framework

The core addition is a DAG-based pipeline framework where each stage runs its own rtp-llm C++ engine instance, orchestrated by a Python-level coordinator.

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                     OmniEngine                              │
                    │                                                             │
  User Request ──►  │  OmniOrchestrator                                           │
                    │      │                                                      │
                    │      ├──► StagePool[0] ──► Pipeline[Thinker]  (C++ AR)      │
                    │      │        │                                              │
                    │      │    StageConnector (shared memory)                     │
                    │      │        │                                              │
                    │      ├──► StagePool[1] ──► Pipeline[Talker]   (C++ AR)      │
                    │      │        │                                              │
                    │      │    StageConnector (shared memory)                     │
                    │      │        │                                              │
                    │      └──► StagePool[2] ──► Pipeline[Code2Wav] (Generation)  │
                    │                                     │                       │
                    │              ◄─── audio waveform ───┘                       │
                    └─────────────────────────────────────────────────────────────┘
```

### 2.2 Key Abstractions

| Abstraction | Source (vllm-omni) | rtp-llm Name | Responsibility |
|---|---|---|---|
| `PipelineConfig` | `vllm_omni.config.stage_config` | `OmniPipelineConfig` | Frozen DAG topology definition |
| `StagePipelineConfig` | `vllm_omni.config.stage_config` | `OmniStageConfig` | Per-stage config (type, inputs, outputs) |
| `StageExecutionType` | `vllm_omni.config.stage_config` | `StageExecutionType` | `LLM_AR`, `LLM_GENERATION`, `DIFFUSION` |
| `Orchestrator` | `vllm_omni.engine.orchestrator` | `OmniOrchestrator` | Request lifecycle across stages |
| `StagePool` | `vllm_omni.engine.stage_pool` | `OmniStagePool` | Manages replicas of one stage |
| `OmniConnectorBase` | `vllm_omni.distributed.omni_connectors` | `StageConnector` | Inter-stage data transfer |
| `OutputProcessor` | `vllm_omni.engine.output_processor` | `OmniOutputProcessor` | Multimodal output assembly |
| `DiffusionEngine` | `vllm_omni.diffusion.diffusion_engine` | `DiffusionEngine` | Non-AR diffusion execution |

### 2.3 Execution Types

Three stage execution types, matching vllm-omni:

- **LLM_AR**: Autoregressive token generation with KV cache. Used for thinker and talker stages. Runs on rtp-llm's existing C++ engine.
- **LLM_GENERATION**: Non-autoregressive LLM-based generation (e.g., code2wav). Uses the C++ engine but without KV cache management and with different scheduling.
- **DIFFUSION**: Iterative denoising (DiT models). Requires a separate diffusion engine with its own scheduler.

## 3. Module Structure

```
rtp_llm/omni/
├── __init__.py
├── config/
│   ├── __init__.py
│   ├── stage_config.py              # StageExecutionType, OmniStageConfig, OmniPipelineConfig
│   ├── pipeline_registry.py         # Central lazy registry of pipeline topologies
│   └── output_modality.py           # OutputModality flag enum (text, audio, image, video)
├── engine/
│   ├── __init__.py
│   ├── omni_engine.py               # OmniEngine: wraps multiple stage engines
│   ├── orchestrator.py              # Request lifecycle, stage-to-stage routing
│   ├── stage_pool.py                # Replica management per stage
│   ├── stage_connector.py           # Inter-stage data transfer (shared memory)
│   ├── stage_processor_base.py      # Base class for inter-stage data transforms
│   └── output_processor.py          # Multimodal output assembly and encoding
├── models/
│   ├── __init__.py
│   ├── qwen2_5_omni/
│   │   ├── __init__.py
│   │   ├── pipeline.py              # QWEN2_5_OMNI_PIPELINE topology
│   │   ├── thinker.py               # Thinker stage model (extends existing Qwen model)
│   │   ├── talker.py                # Talker stage model
│   │   ├── token2wav.py             # Code2Wav stage model
│   │   └── stage_processors.py      # thinker2talker, talker2code2wav transforms
│   └── qwen3_omni/
│       ├── __init__.py
│       ├── pipeline.py              # QWEN3_OMNI_PIPELINE topology
│       ├── thinker.py
│       ├── talker.py
│       ├── code2wav.py
│       └── stage_processors.py
├── diffusion/
│   ├── __init__.py
│   ├── diffusion_engine.py          # Non-AR diffusion execution loop
│   ├── executor.py                  # Model executor for diffusion models
│   ├── scheduler.py                 # Diffusion request scheduling
│   └── registry.py                  # Diffusion model registry
└── serving/
    ├── __init__.py
    ├── openai_omni.py               # OpenAI-compatible omni API extensions
    ├── audio_utils.py               # Audio encoding/decoding (WAV, base64)
    └── protocol.py                  # Omni-specific request/response types
```

## 4. Integration with Existing rtp-llm

### 4.1 Model Registration and Factory

Each stage model registers independently via `register_model()`. The `OmniPipelineConfig` maps HF `architectures` to multi-stage topologies.

```python
# In model_factory.py - detect omni models
class ModelFactory:
    @staticmethod
    def _create_model(model_config, engine_config, ...):
        model_type = model_config.model_type
        
        # Check if this is an omni model with multi-stage pipeline
        pipeline_config = OmniPipelineRegistry.get(model_type)
        if pipeline_config is not None:
            return OmniEngine.from_pipeline_config(
                pipeline_config, model_config, engine_config
            )
        
        # Existing single-model path
        model_cls = ModelFactory.get_model_cls(model_type)
        ...
```

### 4.2 Engine Lifecycle

1. `ModelFactory.create()` detects omni model type via pipeline registry
2. `OmniEngine` reads `OmniPipelineConfig` (e.g., 3 stages for Qwen2.5-Omni)
3. For each stage:
   - Load stage-specific model weights from the HF checkpoint
   - Create a `BaseModel` subclass instance via the stage's registered model class
   - Instantiate a `Pipeline` with its own C++ engine, KV cache, and scheduler
4. `OmniOrchestrator` starts in a background asyncio loop
5. `OmniStagePool` wraps each stage's pipeline for replica management

### 4.3 Request Flow (Qwen2.5-Omni Example)

```
1. OpenAI API: POST /v1/chat/completions
   {messages: [...], modalities: ["text", "audio"]}

2. OmniOrchestrator.submit(request)
   → Create OmniRequestState (tracks progress across stages)

3. Stage 0 (Thinker):
   - Multimodal input preprocessing (images, audio → embeddings via existing MultiModalMixin)
   - Autoregressive text generation with C++ engine
   - Output: text tokens (streamed to client) + hidden state embeddings (latent)
   
4. Stage Processor: thinker2talker()
   - Extract hidden states from thinker output
   - Format as talker input prompt + embeddings

5. Stage 1 (Talker):
   - Receives text embeddings as prompt_embeds
   - Autoregressive speech token generation with C++ engine
   - Output: speech token sequence

6. Stage Processor: talker2code2wav()
   - Convert speech tokens to code2wav input format

7. Stage 2 (Code2Wav):
   - Non-autoregressive waveform generation
   - Output: audio waveform tensor

8. Output Processor:
   - Encode waveform as WAV → base64
   - Assemble final response with text + audio
   - Return to API server
```

### 4.4 Reused Components

- **MultiModalMixin**: Multimodal input processing for thinker stage (images, audio, video)
- **Pipeline + BackendRPCServerVisitor**: Per-stage inference execution
- **ModelFactory + register_model()**: Stage model registration
- **Tokenizer infrastructure**: Per-stage tokenizer support
- **OpenAI renderer framework**: Extended with audio/image output renderers
- **KV cache management**: Per-stage independent KV caches in C++ engine

## 5. Inter-Stage Data Transfer

### 5.1 StageConnector Interface

```python
class StageConnector(ABC):
    def put(self, request_id: str, stage_id: int, data: StageOutput) -> bool:
        """Store stage output for downstream consumption."""
        ...
    
    def get(self, request_id: str, stage_id: int) -> StageOutput | None:
        """Retrieve stage output for next stage input."""
        ...
    
    def cleanup(self, request_id: str) -> None:
        """Release all resources for a completed request."""
        ...
```

### 5.2 StageOutput Data

```python
@dataclass
class StageOutput:
    token_ids: list[int] | None = None
    embeddings: torch.Tensor | None = None       # Hidden state embeddings
    audio_waveform: torch.Tensor | None = None    # Generated audio
    image_tensor: torch.Tensor | None = None      # Generated image
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 5.3 Implementation: SharedMemoryConnector

Initial implementation uses in-process shared memory (Python dict + torch tensors on GPU):
- Zero-copy for GPU tensors when stages run on same device
- Keyed by `(request_id, source_stage_id)`
- Automatic cleanup on request completion

Future: RDMA connector for disaggregated multi-node deployment.

## 6. Diffusion Engine

For image/video generation stages (`StageExecutionType.DIFFUSION`):

### 6.1 Architecture

```python
class DiffusionEngine:
    def __init__(self, model, config):
        self.model = model                    # DiT model
        self.scheduler = DiffusionScheduler() # Request batching
        self.executor = DiffusionExecutor()   # Denoising loop
    
    async def generate(self, request: DiffusionRequest) -> DiffusionOutput:
        """Run iterative denoising to generate image/video."""
        latents = self.initialize_latents(request)
        for step in range(request.num_steps):
            latents = self.executor.denoise_step(latents, step)
        return self.decode_latents(latents)
```

### 6.2 Key Differences from AR Engine

- No KV cache — each step processes the full latent
- Batched execution of multiple denoising steps
- Different scheduling: request-level rather than token-level
- Output: image/video tensor instead of token sequence

## 7. API Extensions

### 7.1 Chat Completions (Extended)

```json
// Request
{
  "model": "Qwen2.5-Omni-7B",
  "messages": [{"role": "user", "content": "Tell me a story"}],
  "modalities": ["text", "audio"],
  "audio": {"voice": "alloy", "format": "wav"}
}

// Response  
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Once upon a time...",
      "audio": {
        "data": "<base64-encoded-wav>",
        "format": "wav"
      }
    }
  }]
}
```

### 7.2 New Endpoints

- `POST /v1/audio/speech`: Dedicated TTS endpoint
- `POST /v1/images/generations`: Image generation endpoint
- Both endpoints follow OpenAI's API format for compatibility

### 7.3 Streaming

- Text output streams immediately via SSE (existing behavior)
- Audio/image output delivered as final chunk when generation completes
- Each stage's output can be independently streamed to the next stage

## 8. Pipeline Configuration (Qwen2.5-Omni Example)

```python
QWEN2_5_OMNI_PIPELINE = OmniPipelineConfig(
    model_type="qwen2_5_omni",
    model_arch="Qwen2_5OmniForConditionalGeneration",
    stages=(
        OmniStageConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            model_cls="Qwen2_5OmniThinker",
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
            model_cls="Qwen2_5OmniTalker",
            input_sources=(0,),
            engine_output_type="latent",
            stage_processor="qwen2_5_omni.thinker2talker",
        ),
        OmniStageConfig(
            stage_id=2,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            model_cls="Qwen2_5OmniToken2Wav",
            input_sources=(1,),
            final_output=True,
            final_output_type="audio",
            stage_processor="qwen2_5_omni.talker2code2wav",
        ),
    ),
)
```

## 9. Implementation Phases

### Phase 1: Core Pipeline Framework
- `OmniPipelineConfig`, `OmniStageConfig`, `StageExecutionType`
- Pipeline registry with lazy loading
- `OmniOrchestrator` with request lifecycle management
- `OmniStagePool` wrapping rtp-llm `Pipeline` instances
- `SharedMemoryConnector` for inter-stage transfer
- `OmniOutputProcessor` for multimodal output assembly

### Phase 2: Qwen2.5-Omni Support
- Thinker stage model (extends existing Qwen architecture)
- Talker stage model
- Token2Wav (Code2Wav) stage model
- Stage processors for inter-stage data transformation
- Weight loading for multi-component HF checkpoints
- E2E testing with Qwen2.5-Omni-7B

### Phase 3: Qwen3-Omni Support
- Qwen3-Omni MoE thinker, talker, code2wav stages
- Reuse pipeline framework from Phase 2
- MoE-specific optimizations

### Phase 4: API and Serving
- OpenAI-compatible API extensions (modalities, audio output)
- `/v1/audio/speech` endpoint
- Audio encoding utilities (WAV, base64)
- Streaming support for multi-stage outputs

### Phase 5: Diffusion Engine
- `DiffusionEngine` for non-AR generation
- `DiffusionExecutor` and `DiffusionScheduler`
- Integration as `StageExecutionType.DIFFUSION` in pipeline
- Image/video generation model support (GLM-Image, Bagel)
- `/v1/images/generations` endpoint

### Phase 6: Distributed Execution
- RDMA-based `StageConnector` for multi-node deployment
- Per-stage device placement and parallelism
- Dynamic replica scaling per stage

## 10. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| C++ engine doesn't support latent/embedding output | Add output mode to C++ engine to return hidden states alongside tokens |
| Weight loading complexity for multi-component checkpoints | Extend `ModelLoader` to load weights per-stage from subdirectories |
| Memory pressure from multiple engines | Support model offloading between stages; lazy engine initialization |
| Stage coordination latency | Shared memory connector minimizes transfer overhead; async overlap |
| Diffusion engine is a large scope addition | Phase it separately; start with AR-only omni models |

## 11. Success Criteria

1. Qwen2.5-Omni-7B generates correct text + audio output via OpenAI API
2. Qwen3-Omni generates correct text + audio output
3. Latency is within 20% of vllm-omni for same models on same hardware
4. Pipeline framework supports adding new omni models without framework changes
5. Existing single-model (non-omni) rtp-llm functionality is unaffected
