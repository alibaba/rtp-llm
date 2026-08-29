"""Multimodal ingress ownership shared by Python engine entry points.

Keep this rule aligned with MMProcessorConfig.h.
"""

from rtp_llm.ops import RoleType, VitSeparation

MULTIMODAL_INGRESS_ROLES = (RoleType.PREFILL, RoleType.PDFUSION)


def owns_multimodal_ingress(engine_config) -> bool:
    """Return whether this rank and role accept multimodal input."""
    return (
        engine_config.parallelism_config.tp_rank == 0
        and engine_config.pd_sep_config.role_type in MULTIMODAL_INGRESS_ROLES
    )


def should_create_local_mm_process_engine(model, engine_config) -> bool:
    """Return whether this process must build a local ViT engine."""
    return (
        model.is_multimodal()
        and model.vit_config.vit_separation == VitSeparation.VIT_SEPARATION_LOCAL
        and owns_multimodal_ingress(engine_config)
    )
