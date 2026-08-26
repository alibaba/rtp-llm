import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main


class _ArgumentGroup:
    def __init__(self):
        self.arguments = {}

    def add_argument(self, flag, **kwargs):
        self.arguments[flag] = kwargs


class _Parser:
    def __init__(self):
        self.group = _ArgumentGroup()

    def add_argument_group(self, _name):
        return self.group


def _load_init_master_group_args():
    util_module_name = "rtp_llm.server.server_args.util"
    original_util_module = sys.modules.get(util_module_name)
    fake_util_module = types.ModuleType(util_module_name)
    fake_util_module.str2bool = bool
    sys.modules[util_module_name] = fake_util_module
    try:
        module_path = Path(__file__).parents[1] / "master_group_args.py"
        spec = importlib.util.spec_from_file_location(
            "master_group_args_under_test", module_path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.init_master_group_args
    finally:
        if original_util_module is None:
            sys.modules.pop(util_module_name, None)
        else:
            sys.modules[util_module_name] = original_util_module


class MasterGroupArgsTest(TestCase):
    def test_fallback_argument_names_follow_configuration_ownership(self):
        parser = _Parser()
        master_config = SimpleNamespace()

        _load_init_master_group_args()(parser, master_config)

        arguments = parser.group.arguments
        fallback_arguments = {
            "--master_client_fallback": (
                "MASTER_CLIENT_FALLBACK",
                "master_client_fallback",
            ),
            "--master_client_fallback_worker_grpc_port_override": (
                "MASTER_CLIENT_FALLBACK_WORKER_GRPC_PORT_OVERRIDE",
                "master_client_fallback_worker_grpc_port_override",
            ),
            "--master_client_fallback_worker_status_port": (
                "MASTER_CLIENT_FALLBACK_WORKER_STATUS_PORT",
                "master_client_fallback_worker_status_port",
            ),
            "--master_client_fallback_candidate_pool_size": (
                "MASTER_CLIENT_FALLBACK_CANDIDATE_POOL_SIZE",
                "master_client_fallback_candidate_pool_size",
            ),
            "--master_client_fallback_cold_candidate_batch_size": (
                "MASTER_CLIENT_FALLBACK_COLD_CANDIDATE_BATCH_SIZE",
                "master_client_fallback_cold_candidate_batch_size",
            ),
            "--master_client_fallback_worker_status_concurrency": (
                "MASTER_CLIENT_FALLBACK_WORKER_STATUS_CONCURRENCY",
                "master_client_fallback_worker_status_concurrency",
            ),
            "--master_client_fallback_worker_status_timeout_ms": (
                "MASTER_CLIENT_FALLBACK_WORKER_STATUS_TIMEOUT_MS",
                "master_client_fallback_worker_status_timeout_ms",
            ),
            "--master_client_fallback_prefill_queue_size_threshold": (
                "MASTER_CLIENT_FALLBACK_PREFILL_QUEUE_SIZE_THRESHOLD",
                "master_client_fallback_prefill_queue_size_threshold",
            ),
            "--master_client_fallback_p2p_hit_discount": (
                "MASTER_CLIENT_FALLBACK_P2P_HIT_DISCOUNT",
                "master_client_fallback_p2p_hit_discount",
            ),
            "--master_client_fallback_cache_affinity_first_max_extra_work_tokens": (
                "MASTER_CLIENT_FALLBACK_CACHE_AFFINITY_FIRST_MAX_EXTRA_WORK_TOKENS",
                "master_client_fallback_cache_affinity_first_max_extra_work_tokens",
            ),
            "--master_client_fallback_outstanding_uncached_tokens_threshold": (
                "MASTER_CLIENT_FALLBACK_OUTSTANDING_UNCACHED_TOKENS_THRESHOLD",
                "master_client_fallback_outstanding_uncached_tokens_threshold",
            ),
            "--master_client_fallback_cache_affinity_first_min_hit_rate": (
                "MASTER_CLIENT_FALLBACK_CACHE_AFFINITY_FIRST_MIN_HIT_RATE",
                "master_client_fallback_cache_affinity_first_min_hit_rate",
            ),
            "--master_client_fallback_flexlb_transport_timeout_ms": (
                "MASTER_CLIENT_FALLBACK_FLEXLB_TRANSPORT_TIMEOUT_MS",
                "master_client_fallback_flexlb_transport_timeout_ms",
            ),
            "--master_client_fallback_discovery_refresh_ms": (
                "MASTER_CLIENT_FALLBACK_DISCOVERY_REFRESH_MS",
                "master_client_fallback_discovery_refresh_ms",
            ),
            "--master_client_fallback_discovery_stale_ms": (
                "MASTER_CLIENT_FALLBACK_DISCOVERY_STALE_MS",
                "master_client_fallback_discovery_stale_ms",
            ),
        }
        for flag, (env_name, field_name) in fallback_arguments.items():
            self.assertEqual(arguments[flag]["env_name"], env_name)
            self.assertIs(arguments[flag]["bind_to"][0], master_config)
            self.assertEqual(arguments[flag]["bind_to"][1], field_name)

        self.assertEqual(
            arguments["--master_kvcm_hot_candidate_pool_size"]["env_name"],
            "MASTER_KVCM_HOT_CANDIDATE_POOL_SIZE",
        )
        self.assertNotIn("--master_kvcm_fallback_enabled", arguments)


if __name__ == "__main__":
    main()
