"""Explicit adaptation to client-record v1; raw Java rows are never relabeled."""
from copy import deepcopy
from typing import Protocol
from schema_contract import matches_schema


class ClientRecordReader(Protocol):
    def snapshot_records(self) -> list: ...


class ClientRecordAdapter:
    def __init__(self, read, decode):
        self._read, self._decode = read, decode

    def snapshot_records(self):
        return [self._decode(row) for row in self._read()]


def read_records(reader, deadline, epoch):
    deadline.check()
    rows = reader.snapshot_records()
    if not isinstance(rows, list) or any(
        not isinstance(row, dict)
        or not matches_schema(row, "client_record_schema_version", 1)
        or row.get("env_epoch") != epoch
        or type(row.get("wire_request_id")) is not int
        or type(row.get("attempt")) is not int for row in rows
    ):
        raise ValueError("invalid or stale client_records v1")
    deadline.check()
    return deepcopy(rows)
