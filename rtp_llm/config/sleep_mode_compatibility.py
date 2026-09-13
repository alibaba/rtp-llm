from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class SleepQuiesceCompatibility:
    """Static prerequisites for matching executor rounds on every sleep rank."""

    world_size: int = 1
    tp_size: int = 1
    dp_size: int = 1
    ep_size: int = 1
    num_layers: int = 1
    expert_num: int = 0
    moe_style: int = 0
    moe_layer_index: tuple[int, ...] = ()
    has_system_prompt: bool = False
    ffn_disaggregate: bool = False


def validate_sleep_quiesce_compatibility(
    *, enable_sleep_mode: bool, compatibility: SleepQuiesceCompatibility
) -> None:
    """Reject unsupported sleep layouts before models or GPU groups are created.

    Applies to both sleep levels. TP ranks share an input broadcast per executor
    call; different DP replicas only share those round boundaries when an active
    MoE layer communicates over the whole world. An EP *size* alone does not
    establish that dependency for dense models.
    """
    if not enable_sleep_mode:
        return

    conflicts: List[str] = []
    if compatibility.has_system_prompt:
        # Rank0's resident-prompt preRun calls bypass the engine-loop tickets;
        # their persistent KV entries also lack a sleep/wake reconstruction path.
        conflicts.append("resident system prompts (multi_task_prompt_tokens)")
    if compatibility.ffn_disaggregate:
        conflicts.append("FFN disaggregate executors with asymmetric rounds")
    if (
        compatibility.tp_size < 1
        or compatibility.dp_size < 1
        or compatibility.world_size != compatibility.tp_size * compatibility.dp_size
        or compatibility.ep_size not in (1, compatibility.world_size)
    ):
        conflicts.append(
            "parallel topology requires world_size=tp_size*dp_size and EP=1 or world_size"
        )

    has_moe_layer = compatibility.expert_num > 0 and (
        (compatibility.moe_style == 1 and compatibility.num_layers > 0)
        or (
            compatibility.moe_style == 2
            and any(
                0 <= layer < compatibility.num_layers
                for layer in compatibility.moe_layer_index
            )
        )
    )
    if compatibility.dp_size > 1 and (
        compatibility.ep_size != compatibility.world_size or not has_moe_layer
    ):
        conflicts.append("DP replicas require an active MoE layer with EP=world_size")
    if conflicts:
        raise ValueError(
            "sleep mode does not support "
            + "; ".join(conflicts)
            + ". Disable sleep mode for this configuration."
        )


@dataclass(frozen=True)
class Level2SleepCompatibility:
    """GPU weight owners that must survive a level-2 sleep/wake cycle."""

    lora_adapter_count: int = 0
    merge_lora: bool = False
    local_multimodal_vit: bool = False
    # A checkpoint-backed propose/draft model (e.g. DSV4 MTP) is a fully
    # independent BaseModel whose own GPU weights are blank-remapped on level-2
    # wake. It is now supported: the draft model's WeightManager is chained onto
    # the main model's (ModelFactory.from_model_configs), so the wake reload fans
    # out and restores the draft weights in place from its checkpoint. Retained
    # here for diagnostics/back-compat but no longer a conflict.
    checkpoint_backed_propose_model: bool = False
    eplb_enabled: bool = False
    redundant_expert: int = 0


def validate_level2_sleep_compatibility(
    *,
    enable_sleep_mode: bool,
    sleep_mode_level: int,
    compatibility: Level2SleepCompatibility,
) -> None:
    """Reject level-2 configurations with GPU weights that cannot be reloaded."""
    if not enable_sleep_mode or sleep_mode_level != 2:
        return

    conflicts: List[str] = []
    if compatibility.lora_adapter_count > 0 and not (
        compatibility.merge_lora and compatibility.lora_adapter_count == 1
    ):
        conflicts.append(
            "unmerged or multiple LoRA adapters "
            f"(count={compatibility.lora_adapter_count}, "
            f"merge_lora={compatibility.merge_lora})"
        )
    if compatibility.local_multimodal_vit:
        conflicts.append("local multimodal ViT")
    # checkpoint_backed_propose_model is intentionally NOT a conflict: the draft
    # model's weights are reloaded on wake via the chained WeightManager reload.
    if compatibility.eplb_enabled:
        conflicts.append("MoE EPLB")
    if compatibility.redundant_expert > 0:
        conflicts.append(
            f"redundant experts (redundant_expert={compatibility.redundant_expert})"
        )

    if conflicts:
        raise ValueError(
            "sleep mode level 2 is incompatible with active GPU weight owners: "
            + "; ".join(conflicts)
            + ". Use sleep mode level 1 instead."
        )


def reject_embedding_sleep(*, enable_sleep_mode: bool, is_embedding: bool) -> None:
    """Reject sleep mode on embedding deployments.

    Sleep/wake_up lifecycle is implemented only for the generate engine
    (EngineBase + SleepLifecycleController). EmbeddingEngine has no lifecycle
    controller, and its backend serves EmbeddingRpcService (ARPC) rather than the
    RpcService stub the lifecycle routes call, so enabling sleep would only expose
    non-functional /sleep, /wake_up endpoints. Reject at config time.
    """
    if enable_sleep_mode and is_embedding:
        raise ValueError(
            "sleep mode is not supported for embedding deployments; "
            "disable enable_sleep_mode for this model."
        )


def reject_dynamic_lora_mutation(
    *, enable_sleep_mode: bool, sleep_mode_level: int
) -> None:
    """Prevent runtime LoRA uploads that level-2 wake cannot reconstruct."""
    if enable_sleep_mode and sleep_mode_level == 2:
        raise ValueError(
            "sleep mode level 2 does not support runtime LoRA add/update/load "
            "because the adapter GPU weights cannot be reconstructed after wake. "
            "Remove adapters or use sleep mode level 1 instead."
        )


def reject_dynamic_weight_update(
    *, enable_sleep_mode: bool, sleep_mode_level: int
) -> None:
    """Prevent runtime weight sync (e.g. RLHF) that level-2 wake would silently revert.

    Level-2 wake restores GPU weights from the original on-disk checkpoint, so any
    in-place weight update pushed at runtime would be discarded on the next wake.
    Level-1 (host backup) captures the updated content and is unaffected.
    """
    if enable_sleep_mode and sleep_mode_level == 2:
        raise ValueError(
            "sleep mode level 2 does not support runtime weight update because "
            "the pushed GPU weights are not in the checkpoint and would be reverted "
            "on wake. Use sleep mode level 1 instead."
        )
