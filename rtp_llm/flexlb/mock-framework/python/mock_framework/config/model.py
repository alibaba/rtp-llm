from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Reference:
    step: str
    path: tuple[str, ...]


@dataclass(frozen=True)
class Expect:
    comparisons: dict[str, Any]


@dataclass(frozen=True)
class CallStep:
    id: str
    describe: str
    interface: str
    args: dict[str, Any]
    run: str = 'foreground'
    within: str | None = None


@dataclass(frozen=True)
class CheckStep:
    id: str
    describe: str
    interface: str
    args: dict[str, Any]
    expect: Expect


@dataclass(frozen=True)
class Phase:
    id: str
    describe: str
    timeout: str
    steps: tuple[CallStep | CheckStep, ...]


@dataclass(frozen=True)
class CompiledCase:
    id: str
    describe: str
    name: str | None
    directory: Path
    environment: dict[str, Any]
    phases: tuple[Phase, ...]
    view: Path | None
