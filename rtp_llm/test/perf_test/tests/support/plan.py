"""Performance framework behavior and failure regressions."""

from rtp_llm.test.perf_test.tests.support.fixtures import fixture_config, fixture_model


def plan_config(backend="local"):
    execution = {"backend": "local", "devices": [2, 3]}
    if backend == "aone":
        execution = {
            "backend": "aone",
            "resource": "daily",
            "gpu_type": "H20",
            "max_parallel": 2,
            "shard_count": 2,
        }
    return fixture_config(
        model=fixture_model(
            label="test-model",
            type="registered_model",
            checkpoint="/models/test",
            topology={"tp": 2, "dp": 1, "pp": 1, "ep": 2},
        ),
        execution=execution,
    )


def plan_cases():
    return [
        {"input_len": 1024, "cache_len": 512, "batch_size": 1},
        {"batch_size": 1, "cache_len": 0, "input_len": 512},
    ]
