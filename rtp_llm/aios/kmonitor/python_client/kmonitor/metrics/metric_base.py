import re
from abc import abstractmethod
from typing import Any, Dict, List

# A kmonitor point is rendered as "name ts value k1=v1 k2=v2 ...", so tags must be
# strings without whitespace or "=". Tags may come from request bodies (e.g. a
# null, numeric or nested "source"), so normalize them before they reach the
# reporter. The mapping is intentionally lossy and many-to-one: whitespace, "="
# and control characters become "_" (so "a b", "a=b" and "a_b" share one
# series), and characters that cannot be encoded as UTF-8 (lone surrogates)
# become "?". Only str/bool/int/float values are kept; anything else (None,
# dict, list, objects) maps to UNKNOWN_TAG_VALUE so a request body cannot create
# an unbounded number of series by sending structured values.
_TAG_BAD_CHARS = re.compile(r"[\s=\x00-\x1f\x7f]")
_TAG_SCALAR_TYPES = (str, bool, int, float)
UNKNOWN_TAG_VALUE = "unknown"


def normalize_tag_value(value: Any, empty: str = UNKNOWN_TAG_VALUE) -> str:
    if not isinstance(value, _TAG_SCALAR_TYPES):
        return empty if empty else UNKNOWN_TAG_VALUE
    text = value if isinstance(value, str) else str(value)
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    text = _TAG_BAD_CHARS.sub("_", text.strip())
    return text if text else empty


def normalize_tags(
    tags: Dict[Any, Any], empty: str = UNKNOWN_TAG_VALUE
) -> Dict[str, str]:
    return {
        normalize_tag_value(k, empty): normalize_tag_value(v, empty)
        for k, v in (tags or {}).items()
    }


class MetricDataPoint(object):
    def __init__(self, value: float, tags: Dict[str, str]):
        self.value = value
        self.tags = tags


class MetricBase(object):
    def __init__(self, name: str, tags: Dict[str, str]):
        super(MetricBase, self).__init__()
        self.name: str = name
        # Keep empty default tag values as-is so existing series keep their identity.
        self.tags: Dict[str, str] = normalize_tags(tags, empty="")

    @abstractmethod
    def report(self, value: float = 1, tags: Dict[str, str] = {}) -> None:
        pass

    @abstractmethod
    def fetch_reported_data(self) -> List[MetricDataPoint]:
        pass
