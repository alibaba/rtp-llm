# rtp-omni Development Roadmap

## Overview

A 4-6 month roadmap to evolve rtp-llm's omni framework from its current state (Qwen2.5-Omni only, linear pipeline, single-process) into a production-ready multi-stage inference engine that supports Qwen3-Omni with multi-GPU, multi-process, streaming inter-stage communication, and C++ engine talker execution.

## Current State

**Supported models:** Qwen2.5-Omni (3-stage: thinker → talker → token2wav)

**Framework capabilities:**
- `OmniStageConfig` / `OmniPipelineConfig`: frozen dataclass config with integer stage IDs and linear `input_sources` topology
- `OmniEngine`: duck-typed to `BaseEngine`, manages stage lifecycle
- `SharedMemoryConnector`: in-process dict with threading lock
- `OmniOrchestrator`: linear request state tracking
- `StageProcessorBase` / `StageProcessorRegistry`: inter-stage transform abstraction
- Talker runs as pure PyTorch forward pass (~560 lines in `talker_inference.py`)

**Key limitations vs sglang-omni:**

| Capability | sglang-omni | rtp-llm omni |
|-----------|-------------|--------------|
| Topology | DAG with `next`, `stream_to`, `wait_for` | Linear chain via `input_sources` |
| Streaming between stages | `stream_to` + `can_accept_stream_before_payload` | Not supported |
| Preprocessing stage | Dedicated CPU stage type | Not modeled |
| Process isolation | Each stage in its own process | Single process |
| TP per stage | `tp_size` on `StageConfig` | Not supported |
| GPU placement | Per-stage GPU assignment | Not modeled |
| Dynamic routing | `resolve_*_next_stages` functions | Not supported |
| Models supported | 10 (omni, TTS, ASR, multimodal) | 1 (Qwen2.5-Omni) |

## Target State

- **Qwen3-Omni** fully supported with 8-stage speech pipeline
- Both thinker and talker run on the rtp C++ engine
- Multi-GPU deployment (thinker GPU 0, talker GPU 1)
- Multi-process stage execution with zero-copy shared memory
- Streaming inter-stage communication (thinker streams to talker for partial start)
- OpenAI-compatible API with streaming text + audio responses
- Per-stage observability and error isolation

## Reference Architecture: sglang-omni

sglang-omni (https://github.com/sgl-project/sglang-omni) supports 10 models across 4 categories. Key patterns referenced in this design:

**Qwen3-Omni (8-stage speech pipeline):**
preprocessing → image_encoder → audio_encoder → mm_aggregate → thinker → talker_ar → code2wav → decode

**StageConfig pattern:** Named stages with `next` (downstream), `stream_to` (streaming targets), `wait_for` (fan-in dependencies), `gpu` (placement), `tp_size` (tensor parallelism), `terminal` (final stage), `factory` (dotted path to executor factory).

**Pipeline variants:** text (6 stages), speech (8 stages, 2 GPUs), speech-colocated (8 stages, 1 GPU).

---

## Phase 1: Framework Upgrades (Months 1-2)

### 1.1 DAG Stage Topology

**Goal:** Replace linear `input_sources: Tuple[int, ...]` with named-stage DAG routing.

**Changes to `OmniStageConfig`:**

```python
@dataclass(frozen=True)
class OmniStageConfig:
    name: str                              # "thinker", "talker_ar", etc.
    execution_type: StageExecutionType
    model_cls: str
    factory: str                           # dotted path to executor factory
    factory_args: Dict[str, Any] = field(default_factory=dict)
    gpu: Optional[int] = None              # GPU placement (None = CPU)
    tp_size: int = 1                       # tensor parallelism
    process: str = "pipeline"              # process name for isolation
    next: Union[str, List[str], None] = None  # downstream stage(s)
    stream_to: List[str] = field(default_factory=list)
    wait_for: List[str] = field(default_factory=list)
    merge_fn: Optional[str] = None         # dotted path for fan-in merge
    project_payload: Dict[str, str] = field(default_factory=dict)
    terminal: bool = False
    can_accept_stream_before_payload: bool = False
    # Legacy fields kept for backward compat during migration
    stage_id: Optional[int] = None
    model_type: Optional[str] = None
    input_sources: Tuple[int, ...] = ()
    final_output: bool = False
    final_output_type: Optional[str] = None
    requires_multimodal_data: bool = False
    engine_output_type: Optional[str] = None
    stage_processor: Optional[str] = None
```

**Topology validation:** The `OmniPipelineConfig.validate()` method checks:
- No duplicate stage names
- All `next`, `stream_to`, `wait_for` references resolve to existing stages
- No self-references
- At least one entry point (stage not referenced by any other stage's `next`)
- At least one terminal stage
- DAG is acyclic

**Migration path:** Qwen2.5-Omni pipeline config migrates from integer stage IDs to named stages. Both old and new config formats work during transition.

### 1.2 Inter-Stage Streaming

**Goal:** Enable stages to emit incremental outputs to downstream stages while still processing.

**`StreamChannel` abstraction:**
- Producer side: `channel.emit(chunk)` sends a partial result
- Consumer side: `channel.recv()` yields chunks as they arrive
- Backpressure: bounded queue with configurable depth
- Thread-safe for single-process; extends to multi-process via shared memory queue

**Integration with `StageConnector`:**
```python
class StageConnector(ABC):
    @abstractmethod
    def put(self, request_id: str, stage_name: str, data: StageOutput) -> bool: ...

    @abstractmethod
    def get(self, request_id: str, stage_name: str) -> Optional[StageOutput]: ...

    @abstractmethod
    def open_stream(self, request_id: str, source: str, target: str) -> StreamChannel: ...

    @abstractmethod
    def cleanup(self, request_id: str) -> None: ...
```

**Partial start support:** When `can_accept_stream_before_payload=True`, the downstream stage begins processing streamed chunks before the upstream stage's final payload arrives. The talker uses this to start generating codec tokens while the thinker is still producing text.

### 1.3 Multi-Process Stage Execution

**Goal:** Each stage runs in its own OS process with its own GPU context.

**Process model:**
- Each unique `process` name in the stage configs gets its own subprocess
- Stages with the same `process` value share a process (colocated)
- The main process (OmniEngine) is the coordinator

**IPC mechanism:**
- **CUDA tensors:** `torch.multiprocessing` with `cuda` sharing (zero-copy for same-node GPU tensors)
- **Signaling:** ZMQ `PUSH/PULL` sockets for lightweight message passing (stage readiness, completion, errors)
- **Shared memory:** `torch.Tensor` allocated in CUDA shared memory for inter-stage tensor transfer

**Process lifecycle:**
- `OmniEngine.start()` spawns stage processes
- Each process loads its model, initializes GPU context, registers with coordinator
- Coordinator routes requests through the DAG
- `OmniEngine.stop()` gracefully shuts down all processes

### 1.4 C++ Engine Talker (external_embeddings)

**Goal:** Run the talker's decoder layers in the rtp C++ engine instead of pure PyTorch.

**Current state:** The `GptModelInputIndex` enum already has `inputEmbeddingsNum`, `inputEmbeddingsSize`, `inputEmbeddingsDtype` entries. The TP sync logic in `tpSyncModelInputs` handles allocation and broadcast. What remains:

**Remaining work:**
1. **Engine forward path:** When `input_embeddings` is present in `GptModelInputs`, the engine's embedding step should use the provided tensor instead of calling `embed_tokens()`. Modify the C++ forward path in the model runner.
2. **Python wrapper:** `compute_talker_external_embeddings()` runs Python-side: `embed(codec) + thinker_hs → thinker_to_talker_proj → [seq_len, hidden_size]`. This tensor is set as `input_embeddings` on the `GptModelInputs`.
3. **Autoregressive loop:** Each step, the Python side computes the next token's external embedding and passes it to the C++ engine for the next forward step. This requires a step-by-step generate API (not the current batch `generate()`).
4. **Weight loading:** Talker projection weights (`thinker_to_talker_proj.weight`, `.bias`) loaded via `Qwen25OmniTalkerWeight` (already implemented).

**Validation:** Bit-exact comparison between PyTorch talker and C++ engine talker on a reference checkpoint.

---

## Phase 2: Qwen3-Omni Model Support (Months 2-4)

### 2.1 Thinker (MoE Architecture)

**Architecture:** `Qwen3OmniMoeForConditionalGeneration` — Mixture-of-Experts thinker.

**rtp-llm base:** `Qwen2Moe` model class already exists. Extend for Qwen3 MoE variant:
- Updated attention patterns (GQA configuration)
- Expert routing changes between Qwen2 and Qwen3 MoE
- `max_seq_len=8192`

**Model class:** `Qwen3OmniThinker` extending the MoE base, registered as `qwen3_omni_thinker`.

### 2.2 Multi-Modal Encoders

**Image encoder:**
- ViT-based, similar to existing Qwen2-VL / Qwen2.5-VL image encoders in rtp-llm
- Reuse existing `QWen2_VL` / `QWen2_5_VL` encoder infrastructure where possible
- GPU-bound stage

**Audio encoder:**
- Whisper-variant with chunked attention (similar to our existing `audio_encoder.py` for Qwen2.5-Omni)
- Sinusoidal position embeddings, stride-2 pooling, projection to thinker hidden size
- GPU-bound stage

**Video handling:**
- Frame extraction from video input
- Reuse image encoder per frame, temporal aggregation

### 2.3 Aggregate Stage

Fan-in stage that waits for preprocessing + all encoder outputs, then merges multimodal features for thinker input.

- `wait_for: ["preprocessing", "image_encoder", "audio_encoder"]`
- `merge_fn: "rtp_llm.omni.models.qwen3_omni.merge.merge_for_thinker"`
- CPU-bound (no GPU needed — just concatenates/pads tensors)

### 2.4 Talker (C++ Engine)

- Runs on separate GPU (`gpu=1` default)
- `max_seq_len=32768` (handles long video prompts)
- **Partial start:** begins generating while thinker still streaming
- **Feedback loop:** `feedback_enabled=True`
- Uses C++ engine with `external_embeddings` from Phase 1.4
- Stop token: `eos_token_id=8294` (codec end token)

### 2.5 Code2Wav Stage

- GPU-bound vocoder converting codec tokens → audio waveform
- Receives streamed codec tokens from talker
- `can_accept_stream_before_payload=True`
- Adapt from existing `token2wav_model.py` (DiT + BigVGAN) or use Qwen3-Omni's specific vocoder

### 2.6 Preprocessing Stage (CPU)

New `StageExecutionType.CPU_EXECUTOR`:
- Tokenizes text input
- Detects multimodal content (image URLs, audio files, video)
- Routes to appropriate encoder stages
- No GPU required

### 2.7 Pipeline Config & Variants

```python
# Text-only (6 stages)
preprocessing → image_encoder → audio_encoder → mm_aggregate → thinker → decode

# Speech (8 stages, 2 GPUs)
preprocessing → image_encoder → audio_encoder → mm_aggregate → thinker → [talker_ar, decode] → code2wav

# Speech-colocated (8 stages, 1 GPU)
Same as speech but thinker and talker on GPU 0, partial start disabled
```

Registered as `qwen3_omni` in `OmniPipelineRegistry` with architecture `Qwen3OmniMoeForConditionalGeneration`.

---

## Phase 3: Production Hardening & API (Months 4-6)

### 3.1 OpenAI-Compatible API

Extend rtp-llm HTTP server:
- `POST /v1/chat/completions` — multimodal input (text, image, audio, video), output text + audio
- `POST /v1/audio/speech` — TTS endpoint (for future TTS model support)
- Streaming SSE with interleaved text and audio chunks
- Audio format: WAV base64-encoded in response, or raw PCM streaming

### 3.2 Benchmarking Suite

**Per-stage metrics:**
- Thinker: TTFT (time to first token), tokens/sec, batch utilization
- Talker: codec tokens/sec, partial start latency savings
- Vocoder: real-time factor (RTF), audio quality metrics

**End-to-end metrics:**
- Input → first audio chunk latency
- Input → full response latency
- Concurrent request throughput

**Comparison framework:** Automated benchmarks against sglang-omni on same hardware (A100/H100), same model checkpoint, same prompts.

### 3.3 Error Handling & Resilience

- **Stage failure isolation:** one stage process crashing doesn't take down others; coordinator detects via ZMQ heartbeat
- **Request timeout per stage:** configurable per-stage timeout, kills request if exceeded
- **Graceful degradation:** if talker/vocoder fails, fall back to text-only response
- **Process restart:** automatic stage process restart on crash (with backoff)

### 3.4 Observability

- **Per-stage metrics:** latency histogram, throughput counter, queue depth gauge, GPU utilization
- **Request tracing:** distributed trace ID propagated across stages (compatible with OpenTelemetry)
- **Structured logging:** each log line includes stage name, request ID, stage latency

---

## Milestones & Success Criteria

| Milestone | Target | Success Criteria |
|-----------|--------|-----------------|
| M1: DAG topology + streaming | Month 1 | Qwen2.5-Omni runs on new DAG config; unit tests pass |
| M2: Multi-process + C++ talker | Month 2 | Qwen2.5-Omni runs multi-process with C++ talker; bit-exact vs PyTorch |
| M3: Qwen3-Omni thinker | Month 3 | Thinker generates text on Qwen3-Omni checkpoint |
| M4: Qwen3-Omni speech | Month 4 | End-to-end text+audio generation on Qwen3-Omni |
| M5: API + benchmarks | Month 5 | OpenAI-compatible API serving, benchmark suite passing |
| M6: Production ready | Month 6 | Error handling, observability, deployment docs complete |

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| C++ engine `input_embeddings` path is more complex than expected | Delays Phase 1.4 | Keep PyTorch talker as fallback; C++ talker is optimization, not blocker |
| Qwen3-Omni MoE weight format differs significantly from Qwen2Moe | Delays Phase 2.1 | Study HuggingFace implementation early; may need custom weight loader |
| Multi-process IPC introduces latency | Degrades Phase 1.3 | Benchmark IPC overhead early; fall back to single-process multi-GPU if needed |
| sglang-omni adds new models during our development | Scope creep | Stick to Qwen3-Omni; framework generality means adding models later is straightforward |

## Out of Scope

- TTS models (Higgs, Qwen3-TTS, MOSS, Voxtral, Fish S2-Pro) — framework will support them but model implementation is deferred
- ASR models (Qwen3 ASR, Whisper) — single-stage, can use existing rtp-llm engine directly
- LLaDA2-Uni — diffusion LLM, fundamentally different execution model
- Ming-Omni — similar to Qwen3-Omni but different architecture; add after Qwen3-Omni proves the framework
