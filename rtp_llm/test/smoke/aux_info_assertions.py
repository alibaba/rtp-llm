import operator
import re
from typing import Any, Dict, List

from smoke.common_def import QueryStatus, SmokeException

_OPS = {
    "eq": operator.eq,
    "ne": operator.ne,
    "ge": operator.ge,
    "gt": operator.gt,
    "le": operator.le,
    "lt": operator.lt,
}


def is_aux_info_only(assertions: Dict[str, Any]) -> bool:
    return assertions.get("mode") == "aux_info_only"


def assert_aux_info_assertions(actual_result: Any, assertions: Dict[str, Any]) -> None:
    fields = assertions.get("fields", {})
    if not isinstance(fields, dict):
        raise SmokeException(
            QueryStatus.VALID_FAILED,
            "aux_info_assertions.fields must be a dict",
        )

    failures: List[str] = []
    for path, expected in fields.items():
        actual = _get_path_value(actual_result, path)
        checks = expected if isinstance(expected, dict) else {"eq": expected}
        for op_name, expect_value in checks.items():
            op = _OPS.get(op_name)
            if op is None:
                raise SmokeException(
                    QueryStatus.VALID_FAILED,
                    f"unsupported aux_info assertion op: {op_name}",
                )
            try:
                satisfied = op(actual, expect_value)
            except TypeError as e:
                failures.append(
                    f"{path}: actual {actual!r} can not be compared with {op_name} {expect_value!r}: {e}"
                )
                continue
            if not satisfied:
                failures.append(
                    f"{path}: actual {actual!r} does not satisfy {op_name} {expect_value!r}"
                )

    if failures:
        raise SmokeException(
            QueryStatus.COMPARE_FAILED,
            "aux_info assertions failed:\n" + "\n".join(failures),
        )


def _get_path_value(root: Any, path: str) -> Any:
    value = root
    for part in path.split("."):
        value = _get_part(value, part)
    return value


def _get_part(value: Any, part: str) -> Any:
    match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?", part)
    if not match:
        raise SmokeException(
            QueryStatus.VALID_FAILED,
            f"invalid aux_info assertion path part: {part}",
        )
    key = match.group(1)
    index = match.group(2)

    if isinstance(value, dict):
        if key not in value:
            raise SmokeException(
                QueryStatus.COMPARE_FAILED,
                f"aux_info assertion path missing key: {key}",
            )
        value = value[key]
    else:
        if not hasattr(value, key):
            raise SmokeException(
                QueryStatus.COMPARE_FAILED,
                f"aux_info assertion path missing attr: {key}",
            )
        value = getattr(value, key)

    if index is not None:
        idx = int(index)
        try:
            value = value[idx]
        except (IndexError, TypeError):
            raise SmokeException(
                QueryStatus.COMPARE_FAILED,
                f"aux_info assertion index out of range: {part}",
            )
    return value
