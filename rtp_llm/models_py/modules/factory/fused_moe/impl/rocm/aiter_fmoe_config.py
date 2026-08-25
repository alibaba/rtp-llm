import hashlib
import importlib.metadata
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_LOGGER = logging.getLogger(__name__)

_SUPPORTED_AITER_VERSION = "0.1.17.dev79+g2570b35f9.d20260623"
_SUPPORTED_DEFAULT_CONFIG_SHA256 = (
    "00a7d76ae7c49760b2bb389d9cd38887713878f139dc49ff3b0af5dc65a6039f"
)
_OVERRIDE_CONFIG = (
    Path(__file__).resolve().parent
    / "data"
    / "qwen35_gfx942_tp4_fp8_tuned_fmoe.csv"
)


@dataclass(frozen=True)
class AiterFmoeConfigStatus:
    applied: bool
    reason: str
    aiter_version: Optional[str] = None


_CONFIG_STATUS: Optional[AiterFmoeConfigStatus] = None


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


def configure_aiter_fmoe_overrides() -> AiterFmoeConfigStatus:
    """Install RTP's additive AITER FMoE config before its first lookup.

    AITER supports merging colon-separated config files.  Keep its stock and
    model-specific configs, then append RTP's narrowly scoped gfx942 override.
    Version and source-config fingerprints deliberately make an AITER upgrade
    require review instead of silently carrying a stale kernel dispatch.
    """

    global _CONFIG_STATUS
    if _CONFIG_STATUS is not None:
        return _CONFIG_STATUS

    try:
        distribution = importlib.metadata.distribution("aiter")
    except importlib.metadata.PackageNotFoundError:
        _CONFIG_STATUS = AiterFmoeConfigStatus(False, "aiter is not installed")
        return _CONFIG_STATUS

    version = distribution.version
    if version != _SUPPORTED_AITER_VERSION:
        _CONFIG_STATUS = AiterFmoeConfigStatus(
            False,
            "unsupported AITER version; review whether the override is still needed",
            version,
        )
        return _CONFIG_STATUS

    default_config = Path(
        distribution.locate_file("aiter/configs/tuned_fmoe.csv")
    ).resolve()
    if not default_config.is_file():
        _CONFIG_STATUS = AiterFmoeConfigStatus(
            False, f"AITER default FMoE config is missing: {default_config}", version
        )
        return _CONFIG_STATUS

    actual_sha256 = _sha256(default_config)
    if actual_sha256 != _SUPPORTED_DEFAULT_CONFIG_SHA256:
        _CONFIG_STATUS = AiterFmoeConfigStatus(
            False,
            "AITER default FMoE config changed; review the RTP override before use "
            f"(expected {_SUPPORTED_DEFAULT_CONFIG_SHA256}, got {actual_sha256})",
            version,
        )
        return _CONFIG_STATUS

    if not _OVERRIDE_CONFIG.is_file():
        _CONFIG_STATUS = AiterFmoeConfigStatus(
            False, f"RTP AITER FMoE override is missing: {_OVERRIDE_CONFIG}", version
        )
        return _CONFIG_STATUS

    existing = os.environ.get("AITER_CONFIG_FMOE")
    if existing:
        existing_paths = {
            str(Path(path).resolve()) for path in existing.split(os.pathsep) if path
        }
        if str(_OVERRIDE_CONFIG) not in existing_paths:
            _CONFIG_STATUS = AiterFmoeConfigStatus(
                False,
                "AITER_CONFIG_FMOE was explicitly set without the RTP override; "
                f"include {_OVERRIDE_CONFIG}",
                version,
            )
            return _CONFIG_STATUS
    else:
        config_paths = [*_stock_fmoe_configs(default_config), _OVERRIDE_CONFIG]
        os.environ["AITER_CONFIG_FMOE"] = os.pathsep.join(map(str, config_paths))

    _CONFIG_STATUS = AiterFmoeConfigStatus(True, "RTP override configured", version)
    _LOGGER.info("Configured RTP AITER FMoE dispatch overrides from %s", _OVERRIDE_CONFIG)
    return _CONFIG_STATUS


def require_aiter_fmoe_overrides_for_qwen35_tp4() -> None:
    status = configure_aiter_fmoe_overrides()
    if status.applied:
        return
    raise RuntimeError(
        "Qwen 35B-A3B TP4 FP8 routed MoE on gfx942 requires the reviewed AITER "
        f"dispatch override, but it is inactive: {status.reason}. "
        "Revalidate the small-batch FMoE kernels and update or remove the override."
    )
