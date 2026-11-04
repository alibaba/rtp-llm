import json
from typing import Mapping


NAVI_SCHED_CONFIG_JSON_ENV = "RTP_LLM_NAVI_SCHED_CONFIG_JSON"


def load_navi_sched_config(environ: Mapping[str, str]):
    raw = environ.get(NAVI_SCHED_CONFIG_JSON_ENV, "").strip()
    if not raw:
        return {}
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{NAVI_SCHED_CONFIG_JSON_ENV} must be valid JSON: {error.msg}"
        ) from error
    if not isinstance(config, dict):
        raise ValueError(f"{NAVI_SCHED_CONFIG_JSON_ENV} must be a JSON object")
    for section in ("master", "agent", "client"):
        if section in config and not isinstance(config[section], dict):
            raise ValueError(
                f"{NAVI_SCHED_CONFIG_JSON_ENV}.{section} must be a JSON object"
            )
    return config
