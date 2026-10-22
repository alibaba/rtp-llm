"""Run offline LRU diagnostics over a streamed token-plan input."""

from analysis.workload_profile import profile
from traffic.token_plan import read_token_plan


def run(path, capacities=(128, 256, 512, 1024)):
    return profile(read_token_plan(path), capacities)
