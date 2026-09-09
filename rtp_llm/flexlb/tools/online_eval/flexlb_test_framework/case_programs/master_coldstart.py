"""Twenty requests with ten concurrent clients immediately after readiness, retaining topology samples and placement evidence."""

from ..case_config import output

METADATA = {
    "id": "master_coldstart",
    "description": "Twenty requests with ten concurrent clients immediately after readiness, "
    "retaining topology samples and placement evidence.",
    "category": "master",
}

PROFILES = ["batch-window", "single-nonbatch", "single-batch", "window-nonbatch"]


def burst_twenty(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "burst",
        "master_request_batch",
        params={
            "target": "single",
            "count": 20,
            "concurrency": 10,
            "request_timeout_s": 15,
            "sample_after_s": 10,
            "coldstart": True,
        },
    )
    case.step(
        "verdict",
        "master_coldstart_check",
        params={"snapshot": output("burst", "snapshot")},
    )
    case.step("cleanup", "teardown")


VARIANTS = {
    "burst_twenty": {
        "build": burst_twenty,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
}
