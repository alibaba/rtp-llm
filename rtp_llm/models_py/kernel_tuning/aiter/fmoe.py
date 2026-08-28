import csv
import hashlib
import importlib.metadata
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rtp_llm.models_py.kernel_tuning.types import KernelTuningStatus

_LOGGER = logging.getLogger(__name__)

AITER_FMOE_GFX942_OVERLAY = (
    "aiter_fmoe_gfx942_cu80_m1_16_h2048_i128_e256_topk8_bf16_fp8pt"
)
_SUPPORTED_AITER_VERSION = "0.1.17.dev79+g2570b35f9.d20260623"
_SUPPORTED_DEFAULT_CONFIG_SHA256 = (
    "00a7d76ae7c49760b2bb389d9cd38887713878f139dc49ff3b0af5dc65a6039f"
)
_OVERLAY_CONFIG = (
    Path(__file__).resolve().parent
    / "configs"
    / "gfx942_cu80_fmoe_m1_16_h2048_i128_e256_topk8_bf16_fp8pt.csv"
)
_DISPATCH_KEY_FIELDS = (
    "cu_num",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
)
_AFFECTED_TOKEN_BUCKETS = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class AiterFmoeWorkloadSignature:
    """Static portion of the AITER FMoE dispatch key.

    Tensor-parallel, expert-parallel, and model identity are intentionally
    absent. They matter only through the local workload passed to AITER.
    """

    gfx: str
    cu_num: int
    model_dim: int
    inter_dim: int
    expert: int
    topk: int
    act_type: str
    dtype: str
    q_dtype_a: str
    q_dtype_w: str
    q_type: str
    use_g1u1: int
    doweight_stage1: int


_AFFECTED_WORKLOAD_SIGNATURES = frozenset(
    {
        AiterFmoeWorkloadSignature(
            gfx="gfx942",
            cu_num=80,
            model_dim=2048,
            inter_dim=128,
            expert=256,
            topk=8,
            act_type="ActivationType.Silu",
            dtype="torch.bfloat16",
            q_dtype_a="torch.float8_e4m3fnuz",
            q_dtype_w="torch.float8_e4m3fnuz",
            q_type="QuantType.per_Token",
            use_g1u1=1,
            doweight_stage1=0,
        )
    }
)

_CONFIG_STATUS: Optional[KernelTuningStatus] = None


def is_affected_aiter_fmoe_signature(
    signature: AiterFmoeWorkloadSignature,
) -> bool:
    return signature in _AFFECTED_WORKLOAD_SIGNATURES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stock_fmoe_configs(default_config: Path) -> list[Path]:
    model_config_dir = default_config.parent / "model_configs"
    model_configs = sorted(
        path
        for path in model_config_dir.glob("*tuned_fmoe*.csv")
        if path.is_file() and "untuned" not in path.name
    )
    return [default_config, *model_configs]


def _dispatch_keys(config: Path) -> set[tuple[str, ...]]:
    with config.open(newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = set(reader.fieldnames or ())
        missing = set(_DISPATCH_KEY_FIELDS) - fieldnames
        if missing:
            raise ValueError(
                f"{config} is missing AITER FMoE dispatch columns: {sorted(missing)}"
            )
        return {
            tuple((row.get(field) or "").strip() for field in _DISPATCH_KEY_FIELDS)
            for row in reader
        }


def _validated_overlay_dispatch_keys(config: Path) -> set[tuple[str, ...]]:
    with config.open(newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = set(reader.fieldnames or ())
        missing = set(_DISPATCH_KEY_FIELDS) - fieldnames
        if missing:
            raise ValueError(
                f"{config} is missing AITER FMoE dispatch columns: {sorted(missing)}"
            )
        rows = list(reader)

    tagged_rows = [row for row in rows if (row.get("_tag") or "").strip()]
    if tagged_rows:
        raise ValueError(
            f"{config} has non-empty _tag values; AITER excludes tagged rows from "
            "normal FMoE dispatch"
        )

    actual_keys = {
        tuple((row.get(field) or "").strip() for field in _DISPATCH_KEY_FIELDS)
        for row in rows
    }
    expected_keys = {
        tuple(
            str(token if field == "token" else getattr(signature, field))
            for field in _DISPATCH_KEY_FIELDS
        )
        for signature in _AFFECTED_WORKLOAD_SIGNATURES
        for token in _AFFECTED_TOKEN_BUCKETS
    }
    if actual_keys != expected_keys or len(rows) != len(expected_keys):
        raise ValueError(
            f"{config} dispatch rows do not match the declared affected workload "
            f"signatures and token buckets {_AFFECTED_TOKEN_BUCKETS}"
        )
    return actual_keys


def _status(
    applied: bool, reason: str, version: Optional[str] = None
) -> KernelTuningStatus:
    return KernelTuningStatus(
        overlay=AITER_FMOE_GFX942_OVERLAY,
        applied=applied,
        reason=reason,
        dependency_version=version,
    )


def configure_aiter_fmoe_overlays() -> KernelTuningStatus:
    """Append reviewed RTP dispatch rows before AITER's first FMoE lookup."""

    global _CONFIG_STATUS
    if _CONFIG_STATUS is not None:
        return _CONFIG_STATUS

    try:
        distribution = importlib.metadata.distribution("aiter")
    except importlib.metadata.PackageNotFoundError:
        _CONFIG_STATUS = _status(False, "aiter is not installed")
        return _CONFIG_STATUS

    version = distribution.version
    if version != _SUPPORTED_AITER_VERSION:
        _CONFIG_STATUS = _status(
            False,
            "unsupported AITER version; review whether the overlay is still needed",
            version,
        )
        return _CONFIG_STATUS

    default_config = Path(
        distribution.locate_file("aiter/configs/tuned_fmoe.csv")
    ).resolve()
    if not default_config.is_file():
        _CONFIG_STATUS = _status(
            False, f"AITER default FMoE config is missing: {default_config}", version
        )
        return _CONFIG_STATUS

    actual_sha256 = _sha256(default_config)
    if actual_sha256 != _SUPPORTED_DEFAULT_CONFIG_SHA256:
        _CONFIG_STATUS = _status(
            False,
            "AITER default FMoE config changed; review the RTP overlay before use "
            f"(expected {_SUPPORTED_DEFAULT_CONFIG_SHA256}, got {actual_sha256})",
            version,
        )
        return _CONFIG_STATUS

    overlay_config = _OVERLAY_CONFIG.resolve()
    if not overlay_config.is_file():
        _CONFIG_STATUS = _status(
            False, f"RTP AITER FMoE overlay is missing: {overlay_config}", version
        )
        return _CONFIG_STATUS

    stock_configs = _stock_fmoe_configs(default_config)
    try:
        overlay_keys = _validated_overlay_dispatch_keys(overlay_config)
        for stock_config in stock_configs:
            duplicate_keys = overlay_keys & _dispatch_keys(stock_config)
            if duplicate_keys:
                _CONFIG_STATUS = _status(
                    False,
                    "AITER stock FMoE configs now overlap the RTP overlay; review "
                    f"and remove or refresh {overlay_config}",
                    version,
                )
                return _CONFIG_STATUS
    except (OSError, ValueError) as error:
        _CONFIG_STATUS = _status(
            False, f"failed to validate AITER FMoE config keys: {error}", version
        )
        return _CONFIG_STATUS

    existing = os.environ.get("AITER_CONFIG_FMOE")
    if existing:
        existing_paths = {
            str(Path(path).resolve()) for path in existing.split(os.pathsep) if path
        }
        if str(overlay_config) not in existing_paths:
            _CONFIG_STATUS = _status(
                False,
                "AITER_CONFIG_FMOE was explicitly set without the RTP overlay; "
                f"include {overlay_config}",
                version,
            )
            return _CONFIG_STATUS
    else:
        config_paths = [*stock_configs, overlay_config]
        os.environ["AITER_CONFIG_FMOE"] = os.pathsep.join(map(str, config_paths))

    _CONFIG_STATUS = _status(True, "RTP AITER FMoE overlay configured", version)
    _LOGGER.info(
        "Configured kernel tuning overlay %s from %s",
        AITER_FMOE_GFX942_OVERLAY,
        overlay_config,
    )
    return _CONFIG_STATUS


def require_aiter_fmoe_tuning(signature: AiterFmoeWorkloadSignature) -> None:
    """Fail closed only for a workload with known-bad stock dispatch rows."""

    if not is_affected_aiter_fmoe_signature(signature):
        return
    status = configure_aiter_fmoe_overlays()
    if status.applied:
        return
    raise RuntimeError(
        "The AITER FMoE workload signature "
        f"{signature} for token buckets {_AFFECTED_TOKEN_BUCKETS} requires the "
        f"reviewed one-stage tuning overlay, but it is inactive: {status.reason}. "
        "Revalidate the small-token FMoE kernels and update or remove the overlay."
    )
