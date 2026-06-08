import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch

from rtp_llm.omni.config.stage_config import (
    OmniPipelineConfig,
    OmniStageConfig,
    StageExecutionType,
)
from rtp_llm.omni.engine.orchestrator import OmniOrchestrator
from rtp_llm.omni.engine.output_processor import OmniOutputProcessor
from rtp_llm.omni.engine.stage_connector import SharedMemoryConnector, StageConnector, StageOutput
from rtp_llm.omni.engine.stage_pool import OmniStagePool

logger = logging.getLogger(__name__)


class OmniEngine:
    """Multi-stage engine for omni models (e.g. thinker → talker → token2wav).

    Implements the same interface as BaseEngine (duck-typed) so it can be
    used by BackendManager without inheritance from BaseEngine (which
    requires a BaseModel in __init__).
    """

    def __init__(
        self,
        pipeline_config: OmniPipelineConfig,
        connector: Optional[StageConnector] = None,
        model_config: Any = None,
        engine_config: Any = None,
    ):
        self.pipeline_config = pipeline_config
        self.config = model_config
        self.model_config = model_config
        self.engine_config = engine_config
        self.connector = connector or SharedMemoryConnector()
        self.output_processor = OmniOutputProcessor()

        self.stage_pools: Dict[int, OmniStagePool] = {}
        for stage_config in pipeline_config.stages:
            self.stage_pools[stage_config.stage_id] = OmniStagePool(
                stage_config=stage_config
            )

        self.orchestrator = OmniOrchestrator(
            pipeline_config=pipeline_config,
            connector=self.connector,
            stage_pools=self.stage_pools,
        )

        self.stage_engines: Dict[int, Any] = {}
        self._primary_engine = None
        self._thinker_engine = None
        self._talker_engine = None
        self.started = False

        self._talker_wrapper = None
        self._token2wav = None
        self._speaker_data: Dict[str, Any] = {}

        logger.info(
            f"OmniEngine created for {pipeline_config.model_type} "
            f"with {len(pipeline_config.stages)} stages"
        )

    @property
    def num_stages(self) -> int:
        return len(self.pipeline_config.stages)

    @property
    def task_type(self):
        if self._primary_engine is not None:
            return self._primary_engine.config.task_type
        from rtp_llm.ops import TaskType
        return TaskType.LANGUAGE_MODEL

    @property
    def default_generate_config(self):
        if self._primary_engine is not None:
            return self._primary_engine.default_generate_config
        return None

    @property
    def role_type(self) -> str:
        if self._primary_engine is not None and hasattr(self._primary_engine, 'role_type'):
            return self._primary_engine.role_type
        return "omni"

    def get_final_output_types(self) -> Dict[str, int]:
        result = {}
        for stage in self.pipeline_config.stages:
            if stage.final_output and stage.final_output_type:
                result[stage.final_output_type] = stage.stage_id
        return result

    def _create_stage_model_config(
        self,
        stage: OmniStageConfig,
        main_model_config: Any,
        engine_config: Any,
    ) -> Any:
        """Create a stage-specific ModelConfig for non-primary stages (e.g. talker).

        Calls model_cls._create_config() for architecture-specific settings,
        then copies shared runtime settings from the main (thinker) config.
        """
        from rtp_llm.model_factory import ModelFactory

        model_cls = ModelFactory.get_model_cls(stage.model_type)
        stage_config = model_cls._create_config(main_model_config.ckpt_path)

        stage_config.ckpt_path = main_model_config.ckpt_path
        stage_config.tokenizer_path = main_model_config.tokenizer_path
        stage_config.model_type = stage.model_type
        stage_config.max_seq_len = main_model_config.max_seq_len
        stage_config.task_type = main_model_config.task_type
        stage_config.use_kvcache = True
        stage_config.phy2log_path = getattr(main_model_config, 'phy2log_path', '')
        stage_config.attn_config.tokens_per_block = (
            main_model_config.attn_config.tokens_per_block
        )
        stage_config.attn_config.kernel_tokens_per_block = (
            main_model_config.attn_config.kernel_tokens_per_block
        )

        if engine_config is not None:
            stage_config.init_precision_config(
                kv_cache_config=engine_config.kv_cache_config, act_type=None
            )
        else:
            stage_config.init_precision_config(kv_cache_config=None, act_type=None)

        logger.info(
            f"Created stage config for {stage.model_stage}: "
            f"hidden_size={stage_config.hidden_size}, "
            f"num_layers={stage_config.num_layers}, "
            f"vocab_size={stage_config.vocab_size}, "
            f"embedding_size={stage_config.embedding_size}"
        )
        return stage_config

    def initialize_stages(
        self,
        model_config: Any,
        engine_config: Any,
        world_info: Any,
        vit_config: Any = None,
        merge_lora: bool = False,
    ) -> None:
        """Create per-stage sub-engines.

        For LLM_AR stages: creates a full LanguageCppEngine via the standard path.
        Each stage gets its own ModelConfig with stage-specific architecture params.
        For LLM_GENERATION stages (token2wav): loaded separately.
        """
        self.model_config = model_config
        self.config = model_config
        self.engine_config = engine_config

        from rtp_llm.model_factory import ModelFactory
        import inspect

        for stage in self.pipeline_config.stages:
            if stage.execution_type == StageExecutionType.LLM_AR:
                logger.info(
                    f"Initializing LLM_AR stage {stage.stage_id} "
                    f"({stage.model_stage}) with model_type={stage.model_type}"
                )

                stage_model_type = stage.model_type
                if stage_model_type is None:
                    raise ValueError(
                        f"Stage {stage.stage_id} ({stage.model_stage}) "
                        f"has no model_type configured"
                    )

                model_cls = ModelFactory.get_model_cls(stage_model_type)

                if self._primary_engine is None:
                    stage_model_config = model_config
                    stage_vit_config = vit_config
                else:
                    stage_model_config = self._create_stage_model_config(
                        stage, model_config, engine_config
                    )
                    stage_vit_config = None

                from_config_kwargs = dict(
                    model_config=stage_model_config,
                    parallelism_config=engine_config.parallelism_config,
                    hw_kernel_config=engine_config.hw_kernel_config,
                    kv_cache_config=engine_config.kv_cache_config,
                    fmha_config=engine_config.fmha_config,
                    moe_config=engine_config.moe_config,
                    load_method=engine_config.load_config.load_method,
                    max_generate_batch_size=engine_config.runtime_config.max_generate_batch_size,
                    vit_config=stage_vit_config,
                    merge_lora=merge_lora,
                    device_resource_config=engine_config.device_resource_config,
                    force_cpu_load_weights=engine_config.load_config.force_cpu_load_weights,
                )
                sig = inspect.signature(model_cls.from_config)
                if 'load_python_model' in sig.parameters:
                    from_config_kwargs['load_python_model'] = True
                if 'skip_python_model' in sig.parameters:
                    from_config_kwargs['skip_python_model'] = False
                stage_model = model_cls.from_config(**from_config_kwargs)

                alog_conf_path = engine_config.profiling_debug_logging_config.ft_alog_conf_path
                from rtp_llm.async_decoder_engine.engine_creator import create_engine
                sub_engine = create_engine(
                    model=stage_model,
                    engine_config=engine_config,
                    alog_conf_path=alog_conf_path,
                    world_info=world_info,
                )
                self.stage_engines[stage.stage_id] = sub_engine

                if self._primary_engine is None:
                    self._primary_engine = sub_engine

                if stage.model_stage == "thinker":
                    self._thinker_engine = sub_engine
                elif stage.model_stage == "talker":
                    self._talker_engine = sub_engine

                logger.info(
                    f"Stage {stage.stage_id} ({stage.model_stage}) engine created"
                )

            elif stage.execution_type == StageExecutionType.LLM_GENERATION:
                logger.info(
                    f"Stage {stage.stage_id} ({stage.model_stage}) "
                    f"is token2wav — will be loaded separately"
                )

            else:
                logger.warning(
                    f"Stage {stage.stage_id} ({stage.model_stage}) "
                    f"has unsupported execution type: {stage.execution_type}"
                )

    def start(self) -> None:
        for stage_id, engine in self.stage_engines.items():
            if hasattr(engine, 'start'):
                stage = self.pipeline_config.get_stage(stage_id)
                logger.info(f"Starting stage {stage_id} ({stage.model_stage})")
                engine.start()
        self.started = True
        logger.info(
            f"OmniEngine started with {len(self.stage_engines)} active stages"
        )

    def stop(self) -> None:
        self.started = False
        for stage_id, engine in self.stage_engines.items():
            if hasattr(engine, 'stop'):
                stage = self.pipeline_config.get_stage(stage_id)
                logger.info(f"Stopping stage {stage_id} ({stage.model_stage})")
                engine.stop()
        logger.info("OmniEngine stopped")

    def ready(self) -> bool:
        if not self.started:
            return False
        for stage_id, engine in self.stage_engines.items():
            if hasattr(engine, 'ready') and not engine.ready():
                return False
        return True

    def initialize_audio_pipeline(self, ckpt_path: str, audio_device: str = "cuda:0"):
        """Load token2wav and speaker data for audio generation.

        The talker engine is already loaded via initialize_stages() as a
        LanguageCppEngine with a Python model. Generation uses the C++ engine's
        autoregressive loop via engine.generate().
        """
        from rtp_llm.omni.models.qwen2_5_omni.token2wav_model import Token2WavModel

        logger.info(f"Loading audio pipeline on {audio_device}...")

        if self._talker_engine is None:
            logger.warning(
                "Talker engine not loaded — call initialize_stages() first"
            )

        self._token2wav = Token2WavModel.from_pretrained(ckpt_path, device=audio_device)

        # Load speaker data
        spk_path = os.path.join(ckpt_path, "spk_dict.pt")
        if os.path.exists(spk_path):
            spk_dict = torch.load(spk_path, map_location=audio_device)
            for name, data in spk_dict.items():
                self._speaker_data[name] = {
                    "bos_token": data["bos_token"],
                    "cond": data["cond"].float().to(audio_device),
                    "ref_mel": data["ref_mel"].float().to(audio_device),
                }
            logger.info(f"Loaded speaker data: {list(self._speaker_data.keys())}")

        self._audio_device = audio_device
        logger.info("Audio pipeline initialized")

    @property
    def audio_pipeline_ready(self) -> bool:
        return (
            self._talker_engine is not None
            and self._token2wav is not None
        )

    def _get_talker_py_model(self):
        if self._talker_engine is None:
            return None
        return getattr(self._talker_engine.model, 'py_model', None)

    @torch.no_grad()
    def generate_audio(
        self,
        text: str,
        per_token_hidden_states: List[List[float]],
        prompt_token_ids: List[int],
        generated_token_ids: List[int],
        speaker: str = "Chelsie",
        max_talker_tokens: int = 4096,
    ) -> Optional[torch.Tensor]:
        """Generate audio waveform from thinker outputs via the C++ engine.

        The talker runs as a LanguageCppEngine. Generation uses the C++ engine's
        autoregressive loop: the Python model's forward() is called at each step,
        combining codec embeddings with thinker hidden states.

        Args:
            text: generated text from thinker
            per_token_hidden_states: list of [hidden_dim] float lists, one per generated token
            prompt_token_ids: token IDs for the input prompt
            generated_token_ids: token IDs for the generated text
            speaker: speaker name
            max_talker_tokens: max codec tokens to generate
        Returns:
            waveform tensor or None on failure
        """
        if not self.audio_pipeline_ready:
            logger.error("Audio pipeline not initialized")
            return None

        if speaker not in self._speaker_data:
            logger.error(f"Unknown speaker: {speaker}. Available: {list(self._speaker_data.keys())}")
            return None

        if not per_token_hidden_states:
            logger.error("No hidden states provided")
            return None

        device = self._audio_device
        dtype = torch.bfloat16
        spk = self._speaker_data[speaker]

        thinker_hs = torch.tensor(
            per_token_hidden_states, dtype=dtype, device=device
        )

        py_model = self._get_talker_py_model()
        if py_model is not None and hasattr(py_model, 'set_thinker_hidden_states'):
            py_model.set_thinker_hidden_states(thinker_hs)

        initial_tokens = torch.tensor(
            [spk["bos_token"]], dtype=torch.int32
        )
        eos_token_id = self._talker_engine.config.special_tokens.eos_token_id

        codec_tokens = self._talker_engine.rtp_llm_op_.generate(
            initial_tokens, max_talker_tokens, eos_token_id
        )

        if py_model is not None and hasattr(py_model, 'clear_thinker_hidden_states'):
            py_model.clear_thinker_hidden_states()

        mask = codec_tokens[0] < 8292
        codec_filtered = codec_tokens[0][mask].unsqueeze(0)
        if codec_filtered.shape[1] == 0:
            logger.warning("All codec tokens were special tokens")
            return None

        logger.info(f"Generated {codec_filtered.shape[1]} codec tokens")

        waveform = self._token2wav(
            codec_filtered.to(device),
            conditioning=spk["cond"],
            reference_mel=spk["ref_mel"],
        )

        logger.info(f"Generated {waveform.numel() / 24000:.2f}s of audio")
        return waveform

    @classmethod
    def from_pipeline_config(
        cls,
        pipeline_config: OmniPipelineConfig,
        model_config: Any = None,
        engine_config: Any = None,
    ) -> "OmniEngine":
        return cls(
            pipeline_config=pipeline_config,
            model_config=model_config,
            engine_config=engine_config,
        )
