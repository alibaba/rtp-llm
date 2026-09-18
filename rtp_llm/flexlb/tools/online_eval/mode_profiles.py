"""Small, validated tables for runtime defaults and master-mode selection.

The resolved address policy concerns network reachability only. Monitoring
identity is the stable (role, engine index, generation), independent of IP.
"""

from __future__ import annotations

import ipaddress
import argparse
import json
import shlex
from pathlib import Path

import yaml

from flexlb_cfg import PROFILE_SPECS, STRESS_PROFILE

TABLE_PATH = Path(__file__).with_name("mode_profiles.yaml")


def load_mode_tables(path: Path = TABLE_PATH) -> dict:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("schema_version") != 1:
        raise ValueError("unsupported mode table schema")
    runtime = doc.get("runtime_modes")
    master = doc.get("master_modes")
    if not isinstance(runtime, dict) or not isinstance(master, dict):
        raise ValueError("both runtime_modes and master_modes are required")
    if set(master) != {"sb", "sn", "wb", "wn"}:
        raise ValueError("master_modes must define sb, sn, wb, wn")
    for name, mode in master.items():
        profile = mode.get("profile")
        if profile not in PROFILE_SPECS:
            raise ValueError(f"{name}: unknown master profile {profile}")
        spec = PROFILE_SPECS[profile]
        if mode.get("decision", "").lower() != spec["decision"] or mode.get(
            "dispatcher", ""
        ).lower() != spec["dispatcher"]:
            raise ValueError(f"{name}: profile axes disagree with flexlb_cfg")
    for name, mode in runtime.items():
        if mode.get("address") not in {
            "loopback_per_engine", "pod_when_external", "pod_per_engine"
        }:
            raise ValueError(f"{name}: invalid address policy")
        obs = mode.get("observation")
        if not isinstance(obs, dict) or set(obs) != {"jsonl", "kmonitor", "report"}:
            raise ValueError(f"{name}: incomplete observation defaults")
        features = mode.get("features")
        if not isinstance(features, dict) or set(features) != {
            "workload", "orchestration", "collector", "assertion"
        }:
            raise ValueError(f"{name}: incomplete feature table")
        if "default_profile" in mode:
            default_mode = mode.get("default_master_mode")
            if mode["default_profile"] != STRESS_PROFILE or default_mode != "wb":
                raise ValueError(f"{name}: unsupported default profile/axis override")
    return doc


def resolve_mode(
    runtime: str,
    master: str,
    *,
    external_engine_clients: bool = False,
    pod_ip: str | None = None,
    unique_loopback_supported: bool = True,
    tables: dict | None = None,
) -> dict:
    """Return an explicit plan; reject an externally unreachable loopback."""
    tables = tables or load_mode_tables()
    if runtime not in tables["runtime_modes"] or master not in tables["master_modes"]:
        raise ValueError(f"unknown runtime/master mode: {runtime}/{master}")
    runtime_cfg = tables["runtime_modes"][runtime]
    master_cfg = tables["master_modes"][master]
    address_plan = resolve_address_plan(
        runtime, external_engine_clients=external_engine_clients,
        pod_ip=pod_ip, tables=tables,
        unique_loopback_supported=unique_loopback_supported,
    )
    selected_profile = (runtime_cfg.get("default_profile")
                        if runtime_cfg.get("default_master_mode") == master
                        else master_cfg["profile"])
    return {
        "runtime": runtime,
        "master_mode": master,
        "master_profile": selected_profile,
        "decision": master_cfg["decision"],
        "dispatcher": master_cfg["dispatcher"],
        **address_plan,
        "metric_identity": "role_engine_index_generation",
        "features": dict(runtime_cfg["features"]),
        "observation": dict(runtime_cfg["observation"]),
    }


def resolve_address_plan(
    runtime: str,
    *,
    external_engine_clients: bool = False,
    pod_ip: str | None = None,
    unique_loopback_supported: bool = True,
    tables: dict | None = None,
) -> dict:
    tables = tables or load_mode_tables()
    if runtime not in tables["runtime_modes"]:
        raise ValueError(f"unknown runtime mode: {runtime}")
    runtime_cfg = tables["runtime_modes"][runtime]
    address = runtime_cfg["address"]
    use_pod = address == "pod_per_engine" or (
        address == "pod_when_external" and external_engine_clients
    )
    if external_engine_clients and not use_pod:
        raise ValueError("external engine clients require a routable engine address")
    if use_pod:
        try:
            parsed = ipaddress.ip_address(pod_ip or "")
        except ValueError as exc:
            raise ValueError("a valid Pod IP is required") from exc
        if parsed.is_loopback or parsed.is_unspecified:
            raise ValueError("a routable Pod IP is required")
    return {
        "advertised_address": "pod_ip" if use_pod else "loopback_per_engine",
        "unique_engine_ips": not use_pod and unique_loopback_supported,
    }


def render_shell_defaults(runtime: str) -> str:
    """Build-time projection for small Whale images without Python/YAML at runtime."""
    tables = load_mode_tables()
    if runtime != "whale_independent":
        raise ValueError("shell defaults are only supported for whale_independent")
    config = tables["runtime_modes"][runtime]
    if config["address"] != "pod_per_engine":
        raise ValueError("independent Whale engines require a Pod address")
    observation = config["observation"]
    if type(observation["jsonl"]) is not bool or type(observation["kmonitor"]) is not bool:
        raise ValueError("Whale shell defaults require boolean output switches")
    values = {
        "MOCK_MODE_RUNTIME": runtime,
        "MOCK_MODE_ADDRESS": config["address"],
        "MOCK_MODE_JSONL_DEFAULT": "1" if observation["jsonl"] else "0",
        "MOCK_MODE_KMONITOR_DEFAULT": "true" if observation["kmonitor"] else "false",
    }
    return "# Generated by mode_profiles.py; do not edit by hand.\n" + "".join(
        f"{key}={shlex.quote(value)}\n" for key, value in values.items()
    )


def master_mode_for_profile(profile: str) -> str:
    if profile == STRESS_PROFILE:
        return "wb"
    matches = [name for name, mode in load_mode_tables()["master_modes"].items()
               if mode["profile"] == profile]
    if len(matches) != 1:
        raise ValueError(f"profile has no unique master mode: {profile}")
    return matches[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Resolve the shared mock runtime/master tables")
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--master")
    parser.add_argument("--profile-to-master")
    parser.add_argument("--pod-ip")
    parser.add_argument("--external-engine-clients", action="store_true")
    parser.add_argument("--field", choices=("master_profile",), default=None)
    parser.add_argument("--render-shell", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.profile_to_master:
            print(master_mode_for_profile(args.profile_to_master))
            return
        if args.render_shell:
            print(render_shell_defaults(args.runtime), end="")
            return
        if not args.master:
            parser.error("--master is required unless --render-shell is used")
        plan = resolve_mode(args.runtime, args.master, pod_ip=args.pod_ip,
                            external_engine_clients=args.external_engine_clients)
    except ValueError as exc:
        parser.error(str(exc))
    print(plan[args.field] if args.field else json.dumps(plan, sort_keys=True))


if __name__ == "__main__":
    main()
