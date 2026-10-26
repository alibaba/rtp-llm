"""MiniMax-M3.1 text model entry on main's configuration and loader contracts."""

import json
import os

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.minimax_m31_config import configure_m31_model, finalize_m31_kv_cache
from rtp_llm.models.minimax_m31_weights import MiniMaxM31Weight
from rtp_llm.ops import RoleType


class MiniMaxM31(BaseModel):
    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        with open(os.path.join(ckpt_path, "config.json")) as reader:
            checkpoint = json.load(reader)
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.model_type = "minimax_m31"
        configure_m31_model(config, checkpoint)
        return config

    @classmethod
    def _apply_kv_cache_config(cls, model_config, kv_cache_config) -> None:
        finalize_m31_kv_cache(model_config)

    def support_cuda_graph(self) -> bool:
        return True

    @staticmethod
    def get_weight_cls():
        return MiniMaxM31Weight

    def _create_python_model(self):
        if (
            self.parallelism_config.role_type == RoleType.PDFUSION
            and self.parallelism_config.dp_size > 1
            and not self.model_config.capture_aux_hidden_layer_ids
        ):
            raise ValueError(
                "MiniMax-M3.1 non-DSpark PDFUSION with DP>1 cannot align Prefill MoE "
                "collectives; use separate Prefill and Decode roles"
            )

        cp = self.parallelism_config.prefill_cp_config
        if (
            self.parallelism_config.role_type in (RoleType.PREFILL, RoleType.PDFUSION)
            and cp.is_enabled()
            and self.parallelism_config.tp_size == 1
        ):
            raise ValueError("MiniMax-M3.1 Prefill CP rotation requires TP>1")

        from rtp_llm.models_py.model_desc.minimax_m31_model import MiniMaxM31Model

        # The M3.1 routed executor processes long Prefill in EP-agreed chunks.
        # Bound only its symmetric buffer; the scheduler admission stays intact.
        prefill_cap = self.model_config.moe_prefill_max_tokens_per_rank
        prefill_limit = (
            16384
            if self.parallelism_config.role_type
            in (RoleType.PREFILL, RoleType.PDFUSION)
            else 8192
        )
        self.model_config.moe_prefill_max_tokens_per_rank = min(
            prefill_limit,
            int(self.model_config.max_seq_len if prefill_cap is None else prefill_cap),
        )
        self.py_model = MiniMaxM31Model(
            self.model_config,
            self.parallelism_config,
            self.weight,
            self.moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model


register_model("minimax_m31", MiniMaxM31)
