from dataclasses import dataclass


@dataclass(frozen=True)
class MaxPanderRuntimeConfig:
    requested: bool = False
    enabled: bool = False
    seed: int = 0
    wf_passes: int = 8
    role: str = "PDFUSION"

    def validate_weight_format(self, *, is_ft_style: bool) -> None:
        if self.enabled and is_ft_style:
            raise ValueError(
                "MaxPander does not support ft-style pre-sharded weights without "
                "expert placement metadata; use the original checkpoint instead"
            )

    def validate_placement(
        self,
        *,
        expert_num: int,
        physical_expert_num: int,
        ep_size: int,
        dynamic_eplb: bool = False,
        phy2log_path: str = "",
    ) -> None:
        if not self.enabled:
            return
        if self.role != "PREFILL":
            raise ValueError("MaxPander is supported only on dedicated PREFILL workers")
        if dynamic_eplb or phy2log_path:
            raise ValueError(
                "MaxPander requires static placement without EPLB or phy2log_path"
            )
        if not (ep_size > 1 and 0 < expert_num <= physical_expert_num <= 2048):
            raise ValueError("MaxPander requires EP>1 and 0<E<=P<=2048")
        if expert_num % ep_size or physical_expert_num % ep_size:
            raise ValueError("MaxPander requires E and P divisible by EP")
        if physical_expert_num > 2 * expert_num:
            raise ValueError("MaxPander supports at most two copies per expert")
        if physical_expert_num // ep_size > 512:
            raise ValueError("MaxPander MegaMoE requires P/EP<=512")

    def to_string(self) -> str:
        return (
            f"requested={self.requested} enabled={self.enabled} role={self.role} "
            f"seed={self.seed} wf_passes={self.wf_passes}"
        )


@dataclass
class MaxPanderConfig:
    enable: bool = False
    seed: int = 0
    wf_passes: int = 8

    def resolve(self, role: str) -> MaxPanderRuntimeConfig:
        role = role.upper()
        if self.enable and role not in ("PREFILL", "DECODE", "FRONTEND"):
            raise ValueError(
                "MaxPander requires dedicated PREFILL/DECODE workers, not PDFUSION or VIT"
            )
        enabled = self.enable and role == "PREFILL"
        if enabled:
            if self.seed < 0:
                raise ValueError("maxpander_seed must be non-negative")
            if not 1 <= self.wf_passes <= 64:
                raise ValueError("maxpander_wf_passes must be in [1,64]")
        return MaxPanderRuntimeConfig(
            requested=self.enable,
            enabled=enabled,
            seed=self.seed,
            wf_passes=self.wf_passes,
            role=role,
        )

    def to_string(self) -> str:
        return f"enable={self.enable} seed={self.seed} wf_passes={self.wf_passes}"
