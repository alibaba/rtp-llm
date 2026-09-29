"""PACE configuration for smoke tests; external services remain caller-owned."""

import json
import os
from pathlib import Path
from urllib.request import urlopen


def runfile(repository, relative):
    root = Path(os.environ["TEST_SRCDIR"])
    repositories = {
        "remote_kv_cache_manager_client_rpm": [
            "remote_kv_cache_manager_client_rpm",
            "remote_kv_cache_manager_client_rpm_cuda130_x86",
            "remote_kv_cache_manager_client_rpm_cuda130_arm",
        ],
        "remote_kv_cache_manager_server": [
            "remote_kv_cache_manager_server",
            "remote_kv_cache_manager_server_cuda130",
        ],
    }.get(repository, [repository])
    candidates = {
        candidate.resolve()
        for repo in repositories
        for candidate in (
            root / repo / relative,
            root / os.environ["TEST_WORKSPACE"] / "external" / repo / relative,
        )
        if candidate.exists()
    }
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one selected runfile: {repository}/{relative}, found {len(candidates)}")
    return candidates.pop()


def require_ok(response):
    code = response.get("header", {}).get("status", {}).get("code")
    if code not in (1, "1", "OK"):
        raise RuntimeError(f"KVCM request failed: {response}")
    return response


def require_binary_path(path):
    try:
        Path(path).resolve().relative_to("/User/gray/tmp")
    except ValueError as error:
        raise RuntimeError("Use a Bazel output_user_root under /User/gray/tmp before running compiled helpers") from error


class PaceFixture:
    def __init__(self, path, backend):
        with open(path, encoding="utf-8") as stream:
            self.config = json.load(stream)
        source_id = os.environ.get("KVCM_EXPECTED_SOURCE_ID", "")
        if not source_id:
            raise RuntimeError("Use a declared P1 smoke target to supply KVCM_EXPECTED_SOURCE_ID")
        if self.config.get("source_id") != source_id:
            raise RuntimeError("PACE fixture source_id does not match the source lock")
        for repo in ("remote_kv_cache_manager_client_rpm", "remote_kv_cache_manager_server"):
            if runfile(repo, "KVCM_SOURCE_ID").read_text().strip() != source_id:
                raise RuntimeError(f"Mismatched binary source marker: {repo}")
        variant = runfile("remote_kv_cache_manager_client_rpm", "KVCM_CLIENT_VARIANT").read_text().strip()
        expected_variant = os.environ.get("KVCM_SMOKE_CLIENT_VARIANT")
        if expected_variant and variant != expected_variant:
            raise RuntimeError(f"This smoke requires a {expected_variant} SDK; select the matching build configuration and artifact")
        if backend not in ("pace", "pace_ssd"):
            raise ValueError("PACE_BACKEND must be pace or pace_ssd")
        domain = self.config.get("domain", "")
        if not domain.startswith(("http://", "https://")):
            raise ValueError("PACE fixture requires an external Meta HTTP URL")
        if not self.config.get("provider_status_urls"):
            raise ValueError("PACE fixture requires provider_status_urls")
        if backend == "pace_ssd" and not self.config.get("ssd_enabled", False):
            raise ValueError("PACE SSD smoke requires an SSD-enabled external Provider")
        self.backend = backend
        self.storage_type = 3 if backend == "pace" else 9
        self.media_type = 2 if backend == "pace" else 5
        self.instance_group = ""

    def check_services(self):
        for url in self.config["provider_status_urls"]:
            with urlopen(url, timeout=10) as response:
                if response.status != 200:
                    raise RuntimeError(f"PACE Provider unavailable: {url}")
                # A byte round trip, rather than this readiness check, proves I/O.
                json.load(response)

    def client_env(self):
        env = dict(self.config.get("client_env", {}))
        if any(not key.startswith(("TAIR_MEMPOOL_", "MC_")) for key in env):
            raise ValueError("PACE client_env only accepts TAIR_MEMPOOL_* and MC_* settings")
        if any(not isinstance(value, str) for value in env.values()):
            raise ValueError("PACE client_env values must be strings")
        env.update({
            "RECO_INSTANCE_GROUP": self.instance_group,
            "RECO_MODEL_SDK_CONFIG": json.dumps([{"type": self.backend}]),
            "RECO_CLIENT_CONFIG": "",
        })
        return env

    def write_startup(self, server_path, work_dir):
        startup = json.loads((Path(server_path) / "etc/default_startup_config.json").read_text())
        # No NFS backend is opened, including during manager bootstrap.
        startup["storage_config"] = {
            "type": self.backend, "global_unique_name": "pace_bootstrap",
            "storage_spec": {"domain": self.config["domain"], "timeout": 10000,
                             "media_type": self.media_type},
        }
        group = startup["instance_group"]
        group["storage_candidates"] = ["pace_bootstrap"]
        group["cache_config"]["cache_prefer_strategy"] = 5 if self.backend == "pace" else 9
        group["cache_config"]["reclaim_strategy"]["storage_unique_name"] = "pace_bootstrap"
        group["quota"] = {"capacity": 1073741824, "quota_config": [
            {"storage_type": self.backend, "capacity": 1073741824},
        ]}
        path = Path(work_dir) / "startup.json"
        path.write_text(json.dumps(startup), encoding="utf-8")
        return str(path)

    def configure(self, server):
        # The manager is private to this test; all allocations use a small quota.
        name = f"pace_smoke_{server._rpc_port}"
        self.instance_group = name
        server.post_json("addStorage", {
            "trace_id": name,
            "storage": {
                "global_unique_name": name,
                "storage_type": self.storage_type,
                "tair_mem_pool": {
                    "domain": self.config["domain"], "timeout": 10000,
                    "media_type": self.media_type,
                },
                "check_storage_available_when_open": True,
            },
        }, admin=True)
        event_name = name + "_events"
        server.post_json("addStorage", {
            "trace_id": name,
            "storage": {
                "global_unique_name": event_name,
                "storage_type": 7,
                "event_report": {
                    "heartbeat_timeout_ms": 30000, "cleanup_grace_ms": 300000,
                    "liveness_check_interval_ms": 5000,
                    "snapshot_min_interval_ms": 1000,
                },
            },
        }, admin=True)
        server.post_json("createInstanceGroup", {
            "trace_id": name,
            "instance_group": {
                "name": name, "storage_candidates": [name],
                "event_report_storage_candidates": [event_name],
                "global_quota_group_name": name, "max_instance_count": 32,
                "quota": {"capacity": 1073741824, "quota_config": [
                    {"storage_type": self.storage_type, "capacity": 1073741824},
                ]},
                "cache_config": {
                    "data_storage_strategy": 5 if self.backend == "pace" else 9,
                    "reclaim_strategy": {
                        "storage_unique_name": name, "reclaim_policy": 1,
                        "trigger_strategy": {"used_percentage": 0.8},
                        "delay_before_delete_ms": 1000,
                    },
                    "meta_indexer_config": {
                        "max_key_count": 10000, "mutex_shard_num": 16,
                        "batch_key_size": 16,
                        "meta_storage_backend_config": {"storage_type": "local"},
                        "meta_cache_policy_config": {"type": "LRU", "capacity": 10000},
                    },
                },
            },
        }, admin=True)
        group = server.post_json("getInstanceGroup", {"name": name}, admin=True)["instance_group"]
        if group.get("storage_candidates") != [name]:
            raise RuntimeError("PACE smoke group must have exactly one payload backend")

    def sdk_config(self, server, instance_id, default_query_type=2):
        return {
            "instance_group": self.instance_group, "instance_id": instance_id,
            "address": [server.address()], "block_size": 16,
            "default_query_type": default_query_type,
            "location_spec_infos": {"full": 262144, "state": 65536},
            "location_spec_groups": {
                "Ffull": ["full"], "Lstate": ["state"],
                "FfullLstate": ["full", "state"],
            },
            "sdk_config": {
                "thread_num": 2, "queue_size": 32, "drain_on_timeout": True,
                "sdk_backend_configs": [{"type": self.backend}],
                "timeout_config": {"put_timeout_ms": 10000, "get_timeout_ms": 10000},
            },
            "model_deployment": {
                "model_name": "pace_contract_smoke", "dtype": "fp16",
                "use_mla": False, "tp_size": 1, "dp_size": 1, "pp_size": 1,
            },
        }
