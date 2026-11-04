"""GLM5.x perf policy; generic runners retain their default behavior."""

from contextlib import contextmanager
from typing import List

from rtp_llm.test.perf_test.batch_perf_impl import BatchPerfImpl
from rtp_llm.test.perf_test.dataclass import ResponseInfo


def normalize_model_paths(argv: List[str]) -> List[str]:
    """Keep the last model-path option so all existing readers agree with argparse."""
    keys = {"--checkpoint_path", "--tokenizer_path", "--sp_checkpoint_path"}
    spans = []
    last = {}
    i = 0
    while i < len(argv):
        key = argv[i].split("=", 1)[0]
        if key in keys:
            end = i + (1 if "=" in argv[i] else 2)
            if end > len(argv):
                raise ValueError(f"Missing value for {key}")
            last[key] = len(spans)
            spans.append((key, argv[i:end]))
            i = end
        else:
            spans.append((None, [argv[i]]))
            i += 1
    return [
        token
        for idx, (key, tokens) in enumerate(spans)
        if key is None or last[key] == idx
        for token in tokens
    ]


def validate_responses(responses: List[ResponseInfo], expected: int) -> None:
    """Never report GLM5.x throughput for an incomplete/failed request batch."""
    successes = sum(bool(response.success) for response in responses)
    if len(responses) != expected or successes != expected:
        raise RuntimeError(
            f"GLM5.x perf request batch failed: {successes}/{expected} requests "
            f"succeeded ({len(responses)} responses received)"
        )


@contextmanager
def validation_policy():
    """Install policy only within the dedicated GLM perf process, then restore it."""
    previous = BatchPerfImpl.__dict__["response_validator"]
    BatchPerfImpl.response_validator = staticmethod(validate_responses)
    try:
        yield
    finally:
        BatchPerfImpl.response_validator = previous
