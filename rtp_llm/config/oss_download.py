import argparse
import json
import os
import re
from functools import lru_cache
from typing import Optional

OSS_INTERNAL_REGION_ALLOWLIST_ENV = "OSS_INTERNAL_REGION_ALLOWLIST"
OSS_REGION_ID_PATTERN = r"[a-z0-9]+(?:-[a-z0-9]+)+"


def parse_oss_internal_region_allowlist(raw: str) -> list[str]:
    try:
        regions = json.loads(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected a JSON array of OSS region IDs"
        ) from error
    if not isinstance(regions, list) or not all(
        isinstance(region, str) and re.fullmatch(OSS_REGION_ID_PATTERN, region)
        for region in regions
    ):
        raise argparse.ArgumentTypeError(
            'expected OSS region IDs such as ["cn-shanghai"]'
        )
    return regions


def export_oss_internal_region_allowlist(regions: Optional[list[str]]) -> None:
    """Pass the resolved CLI/environment policy to all spawned download workers."""
    if regions is not None:
        os.environ[OSS_INTERNAL_REGION_ALLOWLIST_ENV] = json.dumps(regions)


@lru_cache(maxsize=1)
def _region_set(raw: str) -> frozenset[str]:
    return frozenset(parse_oss_internal_region_allowlist(raw))


def get_oss_internal_region_allowlist() -> Optional[frozenset[str]]:
    raw = os.environ.get(OSS_INTERNAL_REGION_ALLOWLIST_ENV)
    return None if raw is None else _region_set(raw)
