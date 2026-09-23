"""capture→fit 行契约的唯一声明（schema 1）。

每行是 JSON object；下列所有字段必须存在。可空字段用 null 表示日志未提供，
不可据此补齐观测值。时间单位 ms，ts 为到达 epoch；il 是精确输入 token 数。
keys 仅包含完整块：SHA256 累计前缀，截取前 128 bit（32 位小写 hex）。
输入按有符号 int32 little-endian 编码；不足 BLOCK_SIZE 的尾块不参与共享匹配。
tail_hash 为全部输入字节的 SHA256 前 128 bit；rid 为来源 ID UTF-8 的同规格摘要。
摘要不含原始 token/ID，无关闭匿名化开关；摘要不是加密，也不保证低熵输入抗猜测。
变更字段或摘要语义必须升级 schema；修改 BLOCK_SIZE 会使旧块大小产物被拒绝。
"""
import math
import re

SCHEMA_VERSION = 1
BLOCK_SIZE = 512
# 类型、可空、含义；本表也驱动运行时校验。
FIELDS = {
    "schema_version": (int, False, "行契约版本"),
    "block_size": (int, False, "完整块 token 数"),
    "ts": (int, False, "request_enter_ts_epoch_ms 到达时间"),
    "il": (int, False, "精确输入 token 数，含残缺尾块"),
    "ol": (int, True, "观测输出 token 数，含错误截断"),
    "keys": (list, False, "完整块的累计前缀摘要"),
    "tail_hash": (str, False, "全部输入的摘要"),
    "rid": (str, False, "upstream_request_id 或 request_id 的摘要"),
    "status": (str, True, "frontend 状态"),
    "error": ((int, str), True, "backend_error_code"),
    "cached": (int, True, "prompt_cached_token_num"),
    "aux_reuse": (int, True, "aux_info.reuse_len"),
    "latency_ms": ((int, float), True, "latency_total_ms"),
    "priority": (int, True, "traffic_reject_priority"),
    "max_new_tokens": (int, True, "生成上限"),
    "timeout_ms": (int, True, "请求超时"),
}
HEX = re.compile(r"[0-9a-f]{32}\Z")


def validate_metadata(value, location):
    for key, expected in (("schema_version", SCHEMA_VERSION), ("block_size", BLOCK_SIZE)):
        if type(value.get(key)) is not int or value[key] != expected:
            raise ValueError(f"{location}: {key} must be {expected}")


def validate_row(row, location="capture row"):
    def fail(reason):
        raise ValueError(f"{location}: {reason}")
    if not isinstance(row, dict):
        fail("expected JSON object")
    for key, (kind, nullable, _) in FIELDS.items():
        if key not in row:
            fail(f"missing required field {key}")
        value = row[key]
        if nullable and value is None:
            continue
        kinds = kind if isinstance(kind, tuple) else (kind,)
        if type(value) not in kinds:
            fail(f"illegal type for {key}")
        if type(value) is float and not math.isfinite(value):
            fail(f"non-finite {key}")
    validate_metadata(row, location)
    if not 1 <= row["il"] <= 2147483647 or row["ts"] < 0:
        fail("invalid il/ts range")
    if len(row["keys"]) != row["il"] // BLOCK_SIZE:
        fail("keys/il block count mismatch")
    for key, values in (("keys", row["keys"]), ("tail_hash", [row["tail_hash"]]), ("rid", [row["rid"]])):
        if any(type(v) is not str or not HEX.fullmatch(v) for v in values):
            fail(f"invalid 128-bit digest in {key}")
    return row
