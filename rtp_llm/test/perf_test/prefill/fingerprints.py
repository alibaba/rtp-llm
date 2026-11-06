"""Stable identity fingerprints shared by prefill initialization and resume."""

import hashlib
from typing import Any, Dict

from rtp_llm.test.perf_test.config_resolver import canonical_json
from rtp_llm.test.perf_test.plan import ExecutionPlan
from rtp_llm.test.perf_test.prefill.measurement import PREFILL_MEASUREMENT_CONTRACT

RUNNER_CONTRACT = {
    "name": "unified-prefill-runner",
    "version": 2,
    "prompt_contract": "exact-token-prefix-v1",
    "measurement_contract": PREFILL_MEASUREMENT_CONTRACT,
}


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def resume_fingerprints(plan: ExecutionPlan) -> Dict[str, str]:
    return {
        "config_fingerprint": plan.config_fingerprint,
        "workload_fingerprint": plan.workload_fingerprint,
        "model_fingerprint": _sha(plan.to_dict()["resolved_config"]["model"]),
        "runner_fingerprint": _sha(RUNNER_CONTRACT),
    }
