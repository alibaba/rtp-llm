"""Validate a parent-owned lane before importing or starting the process harness."""

import os

from .loader import ScenarioError, load_document


def validate_lease(path, budget, environ=None):
    env = os.environ if environ is None else environ
    data = load_document(path)
    required = {
        "schema_version",
        "lane",
        "backend",
        "master_base",
        "mock_base",
        "worker_capacity",
        "intervals",
        "lock_names",
        "child_env",
    }
    if (
        set(data) != required
        or data["schema_version"] != 1
        or data["backend"] != "java_mock"
    ):
        raise ScenarioError("invalid Java mock lease manifest fields or version")
    for key in ("lane", "master_base", "mock_base", "worker_capacity"):
        if type(data[key]) is not int or data[key] < 0:
            raise ScenarioError(f"invalid lease {key}")
    m, b = data["master_base"], data["mock_base"]
    if not (1024 <= m <= 65530 and 1025 <= b <= 65384):
        raise ScenarioError("lease ports outside valid unprivileged range")
    expected = {
        "FLEXLB_FT_MASTER_HTTP_PORT": str(m),
        "FLEXLB_FT_MASTER_MANAGEMENT_PORT": str(m + 1),
        "FLEXLB_FT_HA_MASTER_A_HTTP_PORT": str(m),
        "FLEXLB_FT_HA_MASTER_B_HTTP_PORT": str(m + 3),
        "FLEXLB_FT_MOCK_BASE_GRPC_PORT": str(b),
    }
    if data["child_env"] != expected or any(
        env.get(k) != v for k, v in expected.items()
    ):
        raise ScenarioError("process port environment differs from parent lease")
    intervals = [
        dict(side="master", first=m, last=m + 5),
        dict(side="mock", first=b - 1, last=b + 151),
    ]
    if data["intervals"] != intervals or not (m + 5 < b - 1 or b + 151 < m):
        raise ScenarioError("lease intervals invalid or overlap")
    if data["lock_names"] != [f"m{m}_{m+5}.lock", f"g{b-1}_{b+151}.lock"]:
        raise ScenarioError("lease lock names do not match port windows")
    worker_bound = budget.get("max_environment_workers", budget["initial_workers"])
    if (
        budget["backend"] != "java_mock"
        or budget["bounded"] is not True
        or not 1 <= data["worker_capacity"] <= 149
        or type(worker_bound) is not int
        or worker_bound < budget["initial_workers"]
        or worker_bound + budget["max_dynamic_additions"] > data["worker_capacity"]
    ):
        raise ScenarioError("instance worker budget exceeds lane capacity")
    return data
