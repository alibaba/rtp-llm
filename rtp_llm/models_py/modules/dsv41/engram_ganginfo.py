# SPDX-License-Identifier: Apache-2.0
"""Read the gang member list the platform mounts into every pod.

The pod annotations are mounted at ``/etc/podinfo/annotations`` and C2 lists the
whole gang under ``app.c2.io/biz-detail-ganginfo``::

    app.c2.io/biz-detail-ganginfo="{\"prefill_part0\":{\"name\":\"prefill_part0\",\"ip\":\"10.43.15.156\"},\"prefill_part1\":{\"name\":\"prefill_part1\",\"ip\":\"10.43.5.67\"}}"

Entries carry an address but usually no port, so the shared cohort port applies
to every member.  RTP reads the same file to elect its master
(``rtp_llm.distribute.distributed_server.get_master_from_c2``); this parser is
kept separate because that module imports torch and the Engram startup path must
stay importable without it.
"""
from __future__ import annotations
import json
from pathlib import Path

ANNOTATION_ENV = 'GANG_ANNOCATION_PATH'
DEFAULT_ANNOTATION = '/etc/podinfo/annotations'
ANNOTATION_KEY = 'app.c2.io/biz-detail-ganginfo'


class GangInfoError(ValueError):
    """The annotation is absent, ambiguous or malformed."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def part_number(name):
    """Trailing ``_part<N>`` index, or None when the name has no such suffix."""
    head, separator, tail = name.rpartition('_part')
    if head and separator and tail.isdigit():
        return int(tail)
    return None


def family(name):
    """Gang name shared by every part of one deployment, or None."""
    number = part_number(name)
    return None if number is None else name[:name.rindex('_part')]


def read_parts(path=DEFAULT_ANNOTATION):
    """Return ``[(name, ip, port_or_None)]`` ordered by part number.

    Raises GangInfoError with a stable code instead of leaking OSError/JSON
    errors, so callers can decide whether a missing file is a misconfiguration
    or simply an environment without platform annotations.
    """
    try:
        content = Path(path).read_text()
    except OSError:
        raise GangInfoError('GANGINFO_UNREADABLE') from None
    lines = [line for line in content.splitlines() if ANNOTATION_KEY in line]
    if len(lines) != 1:
        raise GangInfoError('GANGINFO_NOT_UNIQUE')
    line = lines[0].rstrip().replace('\\', '')
    try:
        payload = json.loads(line[line.index('=') + 2:-1])
    except (ValueError, IndexError):
        raise GangInfoError('GANGINFO_INVALID_JSON') from None
    if type(payload) is not dict or not payload:
        raise GangInfoError('GANGINFO_INVALID_JSON')
    parts = []
    for name, info in payload.items():
        if type(info) is not dict or type(info.get('ip')) is not str or not info['ip']:
            raise GangInfoError('GANGINFO_INVALID_JSON')
        port = info.get('port')
        if port in (None, '', 0):
            port = None
        elif type(port) is int and 1 <= port <= 65535:
            pass
        elif type(port) is str and port.isdigit() and 1 <= int(port) <= 65535:
            port = int(port)
        else:
            raise GangInfoError('GANGINFO_INVALID_PORT')
        parts.append((name, info['ip'], port))
    if len({ip for _, ip, _ in parts}) != len(parts):
        raise GangInfoError('GANGINFO_DUPLICATE_ADDRESS')
    if all(part_number(name) is not None for name, _, _ in parts):
        parts.sort(key=lambda item: part_number(item[0]))
    return parts
