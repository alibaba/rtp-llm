"""Resolve Whale placement once; transport flags are implementation details."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys


@dataclass(frozen=True)
class Deployment:
    mode: str
    entrypoint: str
    master_discovery: str
    remote_decode: bool
    colocated_master: bool

    def record(self, runtime, **details):
        plan = {**asdict(self), **details}
        print("Mock deployment: " + json.dumps(plan, sort_keys=True), flush=True)
        Path(runtime, "deployment.json").write_text(json.dumps(plan, indent=2))


MODES = {
    "embedded": Deployment("embedded", "bundle", "file", False, True),
    "single_pod": Deployment("single_pod", "mock", "vipserver", False, False),
    "per_engine": Deployment("per_engine", "mock", "vipserver", True, False),
}


def resolve(env, entrypoint=None, *, warn=True):
    explicit = env.get("MOCK_DEPLOYMENT_MODE")
    role = env.get("MOCK_BUNDLE_ROLE")
    if role is not None and role not in ("bundle", "mock"):
        raise ValueError("MOCK_BUNDLE_ROLE must be bundle or mock")
    remote = env.get("MOCK_REMOTE_DECODE")
    if remote is not None and remote.lower() not in ("0", "1", "false", "true"):
        raise ValueError("MOCK_REMOTE_DECODE must be 0, 1, false or true")
    remote = None if remote is None else remote.lower() in ("1", "true")
    if explicit is None:
        # Compatibility only: old direct invocations keep their entrypoint default.
        selected_role = role or entrypoint or "bundle"
        explicit = ("per_engine" if remote else "single_pod") if selected_role == "mock" else "embedded"
    if explicit not in MODES:
        raise ValueError("MOCK_DEPLOYMENT_MODE must be embedded, single_pod or per_engine")
    plan = MODES[explicit]
    if role is not None and role != plan.entrypoint:
        raise ValueError("MOCK_BUNDLE_ROLE conflicts with MOCK_DEPLOYMENT_MODE")
    if entrypoint is not None and entrypoint != plan.entrypoint:
        raise ValueError("launcher conflicts with MOCK_DEPLOYMENT_MODE")
    if remote is not None and remote != plan.remote_decode:
        raise ValueError("MOCK_REMOTE_DECODE conflicts with MOCK_DEPLOYMENT_MODE")
    file_discovery = env.get("MOCK_BUNDLE_FILE_DISCOVERY")
    if file_discovery is not None and file_discovery != ("1" if plan.colocated_master else "0"):
        raise ValueError("MOCK_BUNDLE_FILE_DISCOVERY conflicts with MOCK_DEPLOYMENT_MODE")
    if not plan.colocated_master:
        for key in ("MOCK_BUNDLE_OVERRIDES_YAML", "MOCK_BUNDLE_CONFIG_PATH", "MOCK_BUNDLE_LEGACY_MASTER"):
            if key in env:
                raise ValueError(f"{key} only applies to embedded deployment")
    for key in ("MOCK_BUNDLE_ROLE", "MOCK_REMOTE_DECODE", "MOCK_BUNDLE_FILE_DISCOVERY"):
        if warn and key in env:
            print(f"Deprecated {key}; use MOCK_DEPLOYMENT_MODE={plan.mode}", file=sys.stderr)
    return plan


def main():
    import os
    plan = resolve(os.environ, warn=False)
    if plan.colocated_master:
        from bundle import run
    elif os.environ.get("ROLE_TYPE") == "VIT":
        from vit import command
        args = command(os.environ)
        os.execv(args[0], args)
        return
    else:
        from standalone_mock import run
    run()


if __name__ == "__main__":
    main()
