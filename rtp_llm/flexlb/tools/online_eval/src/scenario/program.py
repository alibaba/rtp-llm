"""Python 构建器生成的类型化程序；不作为文件格式或配置入口。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from flexlb_cfg import PROFILES


@dataclass(frozen=True)
class StageDefinition:
    id: str
    action: str
    params: dict = field(default_factory=dict)
    timeout_s: Optional[float] = None
    purpose: Optional[str] = None


@dataclass(frozen=True)
class VariantPlan:
    id: str
    stages: tuple[StageDefinition, ...]
    profiles: Optional[tuple[str, ...]] = None
    environment_overrides: dict = field(default_factory=dict)
    execution: dict = field(default_factory=dict)
    metadata: Optional[dict] = None
    reports: Optional[tuple[str, ...]] = None
    reporting: Optional[dict] = None
    monitoring_query_plan: Optional[dict] = None
    requires: tuple[str, ...] = ()
    findings: Optional[tuple[str, ...]] = None


@dataclass(frozen=True)
class ProgramPlan:
    id: str
    description: str
    category: str
    environment: dict
    variants: tuple[VariantPlan, ...]
    profiles: tuple[str, ...] = tuple(PROFILES)
    implementation: dict = field(default_factory=dict)
    execution: dict = field(default_factory=dict)
    requires: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    findings: tuple[str, ...] = ()
    estimated_duration_s: float = 60
