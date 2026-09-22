"""Prometheus text parsing shared by test orchestration and observation adapters."""

from typing import Optional


def _parse_label_block(raw: str) -> dict:
    """Parse the inside of a ``{k1="v1",k2="v2"}`` label block.

    Tolerates the Micrometer/Spring actuator trailing comma
    (``{role="PREFILL",}``).  Label values in this codebase (role /
    engineIp / reason) never carry commas or escapes, so a plain split
    is sufficient — a documented limitation, not a general parser.
    """
    out: dict = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        key, sep, value = pair.partition("=")
        if not sep:
            continue
        out[key.strip()] = value.strip().strip('"')
    return out


def parse_prometheus_samples(
    body: str, name_prefix: str, labels: Optional[dict] = None
) -> list:
    """Parse a Prometheus text-exposition body into prefix-filtered samples.

    Returns ``[(metric_name, labels_dict, value), ...]`` in file order.
    ``# HELP`` / ``# TYPE`` lines, samples whose name does not start with
    ``name_prefix``, samples missing any required ``labels`` pair, and
    lines with unparseable values are skipped; optional trailing
    timestamps are ignored.  Pure function — locally testable with a
    synthetic body, no network involved.
    """
    samples: list = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Metric name runs from column 0 to the first "{" or blank.
        name_end = len(line)
        for idx, ch in enumerate(line):
            if ch == "{" or ch == " ":
                name_end = idx
                break
        name = line[:name_end]
        if not name.startswith(name_prefix):
            continue
        rest = line[name_end:]
        label_values: dict = {}
        if rest.startswith("{"):
            close = rest.find("}")
            if close < 0:
                continue
            label_values = _parse_label_block(rest[1:close])
            rest = rest[close + 1 :]
        if labels and any(label_values.get(k) != v for k, v in labels.items()):
            continue
        parts = rest.split()
        if not parts:
            continue
        try:
            value = float(parts[0])
        except ValueError:
            continue
        samples.append((name, label_values, value))
    return samples
